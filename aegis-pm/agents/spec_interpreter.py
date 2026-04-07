"""
aegis-pm / agents / spec_interpreter.py

Spec Interpreter Agent – parses project charters and PRDs into structured
Jira tasks, then creates them in the configured project.

Architecture
────────────
  SpecParser              Extracts task breakdown from raw text/markdown using LLM
  JiraTaskCreator         Creates Jira issues via REST API with retry
  parse_specification     AutoGen tool – LLM parses PRD → structured task list
  create_jira_task        AutoGen tool – creates one Jira issue, returns key
  build_spec_agents       Returns (AssistantAgent, UserProxyAgent) wired with tools
  SpecInterpreterAgent    High-level class; takes PRD text, returns created keys

AutoGen pattern
───────────────
  UserProxyAgent  (NEVER, no code exec)
    → "Parse this PRD and create Jira tasks for project X"
  AssistantAgent  (GPT-4o, temperature=0.3 for creative decomposition)
    → parse_specification(prd_text)
    → for each task: create_jira_task(...)
    → TERMINATE with summary

Input
─────
  Plain text or Markdown PRD / project charter / feature spec.
  Can be passed as a string or read from a file.

Output
──────
  List of created Jira issue keys: ["ENG-101", "ENG-102", ...]

Usage
─────
  agent = SpecInterpreterAgent()
  keys  = agent.run("## Feature: User Auth\n\n### Requirements\n...")
  print(keys)   # ['ENG-101', 'ENG-102', 'ENG-103']

  # From a file
  keys = agent.run_from_file("docs/prd_user_auth.md")
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from autogen import AssistantAgent, UserProxyAgent, register_function
from dotenv import load_dotenv
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

log = logging.getLogger("aegis.spec_interpreter")


# ── Config ────────────────────────────────────────────────────────────────────

JIRA_BASE_URL  = os.environ["JIRA_BASE_URL"].rstrip("/")
JIRA_EMAIL     = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]
JIRA_PROJECT   = os.environ["JIRA_PROJECT_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Default issue type and priority for created tasks
DEFAULT_ISSUE_TYPE = os.getenv("SPEC_DEFAULT_ISSUE_TYPE", "Story")
DEFAULT_PRIORITY   = os.getenv("SPEC_DEFAULT_PRIORITY", "Medium")
DEFAULT_EPIC_LINK  = os.getenv("SPEC_DEFAULT_EPIC_LINK", "")  # optional epic key

LLM_CONFIG: Dict[str, Any] = {
    "config_list": [
        {
            "model":   os.getenv("OPENAI_MODEL", "gpt-4o"),
            "api_key": OPENAI_API_KEY,
        }
    ],
    "temperature": 0.3,   # slight creativity for good decomposition
    "timeout":     180,   # PRD parsing can take a while
    "cache_seed":  None,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Jira Task Creator
# ══════════════════════════════════════════════════════════════════════════════

class JiraTaskCreator:
    """Creates Jira issues via REST API v3 with tenacity retry."""

    def __init__(self) -> None:
        self._auth    = (JIRA_EMAIL, JIRA_API_TOKEN)
        self._base    = JIRA_BASE_URL
        self._project = JIRA_PROJECT

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=8),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def create_issue(
        self,
        summary:     str,
        description: str,
        issue_type:  str           = DEFAULT_ISSUE_TYPE,
        priority:    str           = DEFAULT_PRIORITY,
        assignee_id: Optional[str] = None,
        labels:      Optional[List[str]] = None,
        story_points: Optional[int]      = None,
        parent_key:  Optional[str]       = None,
    ) -> str:
        """
        Create a single Jira issue.
        Returns the created issue key (e.g. "ENG-42").
        Raises httpx.HTTPStatusError on API errors.
        """
        fields: Dict[str, Any] = {
            "project":   {"key": self._project},
            "summary":   summary,
            "issuetype": {"name": issue_type},
            "priority":  {"name": priority},
            "description": {
                "type":    "doc",
                "version": 1,
                "content": [
                    {
                        "type":    "paragraph",
                        "content": [{"type": "text", "text": description}],
                    }
                ],
            },
        }

        if assignee_id:
            fields["assignee"] = {"id": assignee_id}
        if labels:
            fields["labels"] = labels
        if story_points:
            # Field key varies by Jira config; story_points is the common name
            fields["story_points"] = story_points
        if parent_key:
            fields["parent"] = {"key": parent_key}
        if DEFAULT_EPIC_LINK:
            fields["customfield_10014"] = DEFAULT_EPIC_LINK  # Epic Link field

        url  = f"{self._base}/rest/api/3/issue"
        resp = httpx.post(
            url,
            auth=self._auth,
            json={"fields": fields},
            timeout=20.0,
        )
        resp.raise_for_status()
        issue_key = resp.json()["key"]
        log.info("Jira issue created: %s – %s", issue_key, summary[:60])
        return issue_key

    def build_issue_url(self, key: str) -> str:
        return f"{self._base}/browse/{key}"


_jira_creator = JiraTaskCreator()


# ══════════════════════════════════════════════════════════════════════════════
#  AutoGen Tool Functions
# ══════════════════════════════════════════════════════════════════════════════

def parse_specification(prd_text: str) -> str:
    """
    AutoGen tool: use the LLM to parse a PRD or project charter into a
    structured list of actionable Jira tasks.

    The LLM reads the spec and produces a JSON task breakdown including
    summary, description, type, priority, acceptance criteria, and
    estimated story points for each task.

    Args:
        prd_text: Raw PRD / spec / feature description text (plain text or Markdown)

    Returns:
        JSON string:
        {
          "project_title":  str,
          "total_tasks":    int,
          "tasks": [
            {
              "summary":              str,   max 255 chars, Jira summary line
              "description":          str,   2-4 sentences of context
              "issue_type":           str,   "Story" | "Task" | "Bug" | "Subtask"
              "priority":             str,   "Highest" | "High" | "Medium" | "Low"
              "story_points":         int,   Fibonacci: 1, 2, 3, 5, 8, 13
              "acceptance_criteria":  [str], list of testable acceptance criteria
              "labels":               [str], relevant labels
              "depends_on_index":     int|null  0-based index of a prerequisite task
            }, ...
          ]
        }

    Note: This tool uses an internal LLM call (not the outer AutoGen chat).
    The result is deterministic JSON the outer agent can act on.
    """
    log.info("parse_specification: parsing PRD (%d chars)", len(prd_text))

    # Inner LLM call to parse the spec
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = """You are a senior engineering project manager and Jira expert.
Your job is to read a project specification or PRD and decompose it into
well-formed, actionable Jira tickets.

Rules:
- Each task must be independently completable by one engineer in ≤1 sprint
- Summaries must be concise (max 80 chars) and start with a verb: "Implement", "Add", "Fix", "Create", "Migrate", "Write"
- Descriptions add context without repeating the summary
- Story points use Fibonacci: 1 (trivial), 2 (small), 3 (medium), 5 (large), 8 (very large), 13 (epic – should be split)
- Issue types: "Story" for user-facing features, "Task" for technical work, "Bug" for known defects
- Priority: "Highest" only for P0 issues; default to "Medium"
- Labels should be lowercase: e.g. ["backend", "auth", "database"]
- Accept criteria must be testable: "Given X, when Y, then Z"
- Output ONLY valid JSON matching the exact schema. No markdown, no explanation."""

    user_prompt = f"""Parse this specification and return a JSON task breakdown:

---
{prd_text}
---

Return JSON matching exactly this schema:
{{
  "project_title": "string",
  "total_tasks": number,
  "tasks": [
    {{
      "summary": "string (max 80 chars, starts with verb)",
      "description": "string (2-4 sentences)",
      "issue_type": "Story|Task|Bug|Subtask",
      "priority": "Highest|High|Medium|Low|Lowest",
      "story_points": 1|2|3|5|8|13,
      "acceptance_criteria": ["string"],
      "labels": ["string"],
      "depends_on_index": null|number
    }}
  ]
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)

        # Validate and sanitise
        tasks = data.get("tasks", [])
        for task in tasks:
            # Enforce summary length
            if len(task.get("summary", "")) > 255:
                task["summary"] = task["summary"][:252] + "..."
            # Ensure required fields
            task.setdefault("issue_type",  DEFAULT_ISSUE_TYPE)
            task.setdefault("priority",    DEFAULT_PRIORITY)
            task.setdefault("story_points", 3)
            task.setdefault("labels", [])
            task.setdefault("acceptance_criteria", [])
            task.setdefault("depends_on_index", None)

        data["total_tasks"] = len(tasks)
        log.info(
            "parse_specification: produced %d task(s) for '%s'",
            len(tasks), data.get("project_title", "?"),
        )
        return json.dumps(data)

    except json.JSONDecodeError as exc:
        log.error("parse_specification: LLM returned invalid JSON: %s", exc)
        return json.dumps({"error": f"JSON parse error: {exc}", "tasks": []})
    except Exception as exc:
        log.error("parse_specification: unexpected error: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc), "tasks": []})


def create_jira_task(
    summary:             str,
    description:         str,
    issue_type:          str            = DEFAULT_ISSUE_TYPE,
    priority:            str            = DEFAULT_PRIORITY,
    story_points:        Optional[int]  = None,
    labels:              Optional[List[str]] = None,
    acceptance_criteria: Optional[List[str]] = None,
) -> str:
    """
    AutoGen tool: create a single Jira issue from parsed task data.

    Args:
        summary:             Issue title (max 255 chars, starts with a verb)
        description:         Context and details for the developer
        issue_type:          "Story" | "Task" | "Bug" | "Subtask"
        priority:            "Highest" | "High" | "Medium" | "Low" | "Lowest"
        story_points:        Fibonacci estimate (1, 2, 3, 5, 8, 13)
        labels:              List of lowercase label strings
        acceptance_criteria: List of testable AC strings (appended to description)

    Returns:
        JSON string:
          {"success": true,  "issue_key": "ENG-42", "jira_url": "https://..."}
        or
          {"success": false, "error": "..."}
    """
    # Enrich description with acceptance criteria if provided
    full_description = description
    if acceptance_criteria:
        ac_text = "\n\nAcceptance Criteria:\n" + "\n".join(
            f"- {criterion}" for criterion in acceptance_criteria
        )
        full_description += ac_text

    log.info(
        "create_jira_task: creating %s '%s' (priority=%s, points=%s)",
        issue_type, summary[:60], priority, story_points,
    )

    try:
        issue_key = _jira_creator.create_issue(
            summary=summary,
            description=full_description,
            issue_type=issue_type,
            priority=priority,
            labels=labels or [],
            story_points=story_points,
        )
        return json.dumps({
            "success":   True,
            "issue_key": issue_key,
            "jira_url":  _jira_creator.build_issue_url(issue_key),
        })
    except RetryError as exc:
        msg = f"Jira API unreachable after retries: {exc}"
        log.error("create_jira_task: %s", msg)
        return json.dumps({"success": False, "error": msg})
    except httpx.HTTPStatusError as exc:
        msg = f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
        log.error("create_jira_task: %s", msg)
        return json.dumps({"success": False, "error": msg})
    except Exception as exc:
        log.error("create_jira_task: unexpected error: %s", exc, exc_info=True)
        return json.dumps({"success": False, "error": str(exc)})


# ══════════════════════════════════════════════════════════════════════════════
#  AutoGen Agent Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_spec_agents() -> tuple[AssistantAgent, UserProxyAgent]:
    """
    Build the Spec Interpreter AutoGen agent pair.

    Tool call sequence per run:
      1. parse_specification(prd_text)   → structured task JSON
      2. For each task in order:
           create_jira_task(...)         → Jira issue key
      3. TERMINATE with JSON summary
    """
    assistant = AssistantAgent(
        name="SpecInterpreter",
        llm_config=LLM_CONFIG,
        system_message="""You are the Spec Interpreter for Aegis PM – an autonomous AI project management system.

Your role is to transform vague or detailed project specifications into concrete, well-formed Jira tickets that an engineering team can immediately act on.

Each run:
1. Call `parse_specification` with the PRD text to get a structured task breakdown.
2. Review the returned task list. If any task has story_points=13, consider whether it should be split (but do not re-parse – proceed with what you have).
3. For EACH task in the parsed list, call `create_jira_task` with the task's fields.
   - Respect the `depends_on_index` field: create tasks in order so dependencies are created first.
   - Include acceptance_criteria in each call.
4. After all tasks are created, produce a final JSON summary and end with TERMINATE.

Rules:
- Create every task. Never skip.
- If create_jira_task returns success=false, record the failure but continue.
- Do NOT modify task summaries unless they exceed 255 chars.
- Do NOT ask clarifying questions.
- Your final message MUST end with TERMINATE.

Final summary format:
```json
{
  "spec_summary": {
    "project_title": "<str>",
    "tasks_parsed":  <int>,
    "tasks_created": <int>,
    "tasks_failed":  <int>,
    "created_issues": [
      {"index": 0, "summary": "...", "issue_key": "ENG-101", "jira_url": "..."},
      ...
    ],
    "failed_tasks": [
      {"index": <int>, "summary": "...", "error": "..."}
    ]
  }
}
```
TERMINATE""",
    )

    user_proxy = UserProxyAgent(
        name="SpecOrchestrator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=60,   # large PRDs may have many tasks
        code_execution_config=False,
        is_termination_msg=lambda msg: (
            isinstance(msg.get("content"), str)
            and "TERMINATE" in msg["content"]
        ),
    )

    # parse_specification calls OpenAI internally — the AssistantAgent decides
    # WHEN to call it; the UserProxyAgent executes the function.
    register_function(
        parse_specification,
        caller=assistant,
        executor=user_proxy,
        name="parse_specification",
        description=(
            "Parse a PRD or project spec into a structured JSON task breakdown. "
            "Call this once with the full spec text. Returns tasks with summaries, "
            "descriptions, priorities, story points, and acceptance criteria."
        ),
    )

    register_function(
        create_jira_task,
        caller=assistant,
        executor=user_proxy,
        name="create_jira_task",
        description=(
            "Create a single Jira issue in the configured project. "
            "Pass summary, description, issue_type, priority, story_points, "
            "labels, and acceptance_criteria. Returns the created issue key."
        ),
    )

    return assistant, user_proxy


# ══════════════════════════════════════════════════════════════════════════════
#  High-level SpecInterpreterAgent
# ══════════════════════════════════════════════════════════════════════════════

class SpecInterpreterAgent:
    """
    High-level wrapper. Takes PRD text → creates Jira tasks → returns issue keys.
    """

    def run(self, prd_text: str) -> Dict[str, Any]:
        """
        Parse a PRD and create Jira issues for every task.

        Args:
            prd_text: Plain text or Markdown PRD / project charter

        Returns:
            {
              "project_title":  str,
              "tasks_parsed":   int,
              "tasks_created":  int,
              "tasks_failed":   int,
              "issue_keys":     ["ENG-101", "ENG-102", ...],
              "created_issues": [{...}],
              "failed_tasks":   [{...}],
            }
        """
        if not prd_text.strip():
            return {"error": "Empty PRD text provided", "issue_keys": []}

        log.info(
            "SpecInterpreterAgent: parsing PRD (%d chars) for project %s",
            len(prd_text), JIRA_PROJECT,
        )

        assistant, user_proxy = build_spec_agents()

        task_prompt = (
            f"Parse the following PRD and create Jira tasks in project '{JIRA_PROJECT}'.\n\n"
            f"--- PRD START ---\n{prd_text}\n--- PRD END ---\n\n"
            f"Parse the specification, then create every task in Jira. "
            f"Summarise results and TERMINATE."
        )

        try:
            user_proxy.initiate_chat(assistant, message=task_prompt, silent=True)
        except Exception as exc:
            log.exception("SpecInterpreterAgent chat raised: %s", exc)
            return {"error": str(exc), "issue_keys": []}

        return self._parse_summary(user_proxy.chat_messages.get(assistant, []))

    def run_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read a PRD from a file and run the interpreter.

        Args:
            file_path: Path to a .md, .txt, or .rst file

        Returns:
            Same as run()
        """
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}", "issue_keys": []}
        text = path.read_text(encoding="utf-8")
        log.info("SpecInterpreterAgent: loaded PRD from %s (%d chars)", file_path, len(text))
        return self.run(text)

    @staticmethod
    def _parse_summary(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        for msg in reversed(messages):
            content = msg.get("content") or ""
            if "spec_summary" not in content:
                continue
            start = content.find("{")
            end   = content.rfind("}") + 1
            if start == -1 or end <= start:
                continue
            try:
                data  = json.loads(content[start:end])
                inner = data.get("spec_summary", data)
                return {
                    "project_title":  inner.get("project_title", ""),
                    "tasks_parsed":   inner.get("tasks_parsed", 0),
                    "tasks_created":  inner.get("tasks_created", 0),
                    "tasks_failed":   inner.get("tasks_failed", 0),
                    "issue_keys":     [
                        t["issue_key"]
                        for t in inner.get("created_issues", [])
                        if t.get("issue_key")
                    ],
                    "created_issues": inner.get("created_issues", []),
                    "failed_tasks":   inner.get("failed_tasks", []),
                }
            except json.JSONDecodeError:
                log.warning("Could not parse spec_summary JSON")
                continue

        log.warning("No spec_summary found in chat history")
        return {"tasks_parsed": 0, "tasks_created": 0, "issue_keys": []}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level="INFO", format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser(description="Aegis PM – Spec Interpreter")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file",  help="Path to PRD file (.md, .txt)")
    group.add_argument("--text",  help="Inline PRD text string")
    group.add_argument("--stdin", action="store_true", help="Read PRD from stdin")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse only – do not create Jira issues (prints tasks)")
    args = parser.parse_args()

    if args.stdin:
        prd = sys.stdin.read()
    elif args.file:
        prd = Path(args.file).read_text(encoding="utf-8")
    else:
        prd = args.text

    if args.dry_run:
        print("DRY RUN – parsing only, no Jira issues will be created\n")
        result_json = parse_specification(prd)
        parsed = json.loads(result_json)
        print(f"Project: {parsed.get('project_title', '?')}")
        print(f"Tasks:   {parsed.get('total_tasks', 0)}\n")
        for i, task in enumerate(parsed.get("tasks", [])):
            print(f"  [{i}] [{task['issue_type']:6}] [{task['priority']:7}] "
                  f"[{task['story_points']}pt] {task['summary']}")
        sys.exit(0)

    agent  = SpecInterpreterAgent()
    result = agent.run_from_file(args.file) if args.file else agent.run(prd)

    print("\n" + "═" * 60)
    print("  SPEC INTERPRETER – RESULTS")
    print("═" * 60)
    print(f"  Project:  {result.get('project_title', '?')}")
    print(f"  Parsed:   {result.get('tasks_parsed', 0)} tasks")
    print(f"  Created:  {result.get('tasks_created', 0)} Jira issues")
    print(f"  Failed:   {result.get('tasks_failed', 0)}")
    print()
    for issue in result.get("created_issues", []):
        print(f"  ✓ {issue['issue_key']}  {issue.get('summary', '')[:60]}")
        print(f"    {issue.get('jira_url', '')}")
    if result.get("failed_tasks"):
        print()
        for ft in result["failed_tasks"]:
            print(f"  ✗ [{ft['index']}] {ft.get('summary', '')[:60]}")
            print(f"    Error: {ft.get('error', '')[:100]}")
