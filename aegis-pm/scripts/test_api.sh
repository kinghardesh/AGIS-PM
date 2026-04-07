#!/usr/bin/env bash
# ============================================================
#  Aegis PM – API Smoke Tests  (v2)
#  Usage: bash scripts/test_api.sh [base_url]
# ============================================================

BASE=${1:-"http://localhost:8000"}
OK=0; FAIL=0

banner() { echo; echo "─────── $1 ───────"; }
pass()   { echo "  ✅  $1"; ((OK++)); }
fail()   { echo "  ❌  $1"; ((FAIL++)); }
json_val() { python3 -c "import sys,json; d=json.load(sys.stdin); print(d$2)" 2>/dev/null; }

banner "Health + DB check"
RES=$(curl -s "$BASE/health")
echo "$RES" | grep -q '"ok"\|"degraded"' && pass "GET /health" || fail "GET /health → $RES"

banner "Stats (empty)"
RES=$(curl -s "$BASE/stats")
echo "$RES" | grep -q '"total"' && pass "GET /stats" || fail "GET /stats → $RES"

banner "Create alert"
BODY='{"task_key":"ENG-1","task_summary":"Fix login bug","assignee":"Jane Dev","assignee_email":"jane@example.com","jira_url":"https://jira.example.com/browse/ENG-1"}'
RES=$(curl -s -X POST "$BASE/alerts" -H 'Content-Type: application/json' -d "$BODY")
ID=$(echo "$RES" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null)
[ -n "$ID" ] && pass "POST /alerts → id=$ID" || fail "POST /alerts → $RES"

banner "Create second alert"
BODY2='{"task_key":"ENG-2","task_summary":"Refactor auth","assignee":"Bob Dev"}'
RES2=$(curl -s -X POST "$BASE/alerts" -H 'Content-Type: application/json' -d "$BODY2")
ID2=$(echo "$RES2" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null)
[ -n "$ID2" ] && pass "POST /alerts (second) → id=$ID2" || fail "POST /alerts second → $RES2"

banner "Duplicate suppression"
DUP=$(curl -s -X POST "$BASE/alerts" -H 'Content-Type: application/json' -d "$BODY")
DUP_ID=$(echo "$DUP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null)
[ "$DUP_ID" = "$ID" ] && pass "Duplicate suppressed (same id=$ID)" || fail "Duplicate not suppressed – got id=$DUP_ID vs $ID"

banner "GET /alerts (list all)"
RES=$(curl -s "$BASE/alerts")
echo "$RES" | grep -q '"items"' && pass "GET /alerts returns paginated shape" || fail "GET /alerts → $RES"

banner "GET /alerts?status=pending"
RES=$(curl -s "$BASE/alerts?status=pending")
COUNT=$(echo "$RES" | python3 -c "import sys,json; print(json.load(sys.stdin)['total'])" 2>/dev/null)
[ "$COUNT" -ge 2 ] 2>/dev/null && pass "GET /alerts?status=pending → total=$COUNT" || fail "Expected >= 2 pending, got $COUNT"

banner "GET /alerts?assignee=jane"
RES=$(curl -s "$BASE/alerts?assignee=jane")
echo "$RES" | grep -q '"Jane Dev"' && pass "GET /alerts?assignee=jane (case-insensitive)" || fail "Assignee filter → $RES"

banner "GET /alerts?task_key=ENG-1"
RES=$(curl -s "$BASE/alerts?task_key=ENG-1")
echo "$RES" | grep -q 'ENG-1' && pass "GET /alerts?task_key=ENG-1" || fail "task_key filter → $RES"

banner "GET /alerts?order_by=assignee&order_dir=asc"
RES=$(curl -s "$BASE/alerts?order_by=assignee&order_dir=asc")
echo "$RES" | grep -q '"items"' && pass "GET /alerts with custom sort" || fail "Custom sort → $RES"

banner "GET /alerts (invalid status → 400)"
CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/alerts?status=invalid_status")
[ "$CODE" = "400" ] && pass "Invalid status rejected (400)" || fail "Expected 400, got $CODE"

banner "GET /stats (after creates)"
RES=$(curl -s "$BASE/stats")
TOTAL=$(echo "$RES" | python3 -c "import sys,json; print(json.load(sys.stdin)['total'])" 2>/dev/null)
[ "$TOTAL" -ge 2 ] 2>/dev/null && pass "GET /stats → total=$TOTAL" || fail "Stats total=$TOTAL"

banner "GET /alerts/{id}"
RES=$(curl -s "$BASE/alerts/$ID")
echo "$RES" | grep -q '"pending"' && pass "GET /alerts/$ID → pending" || fail "GET /alerts/$ID → $RES"

banner "Approve alert"
RES=$(curl -s -X POST "$BASE/alerts/$ID/approve" \
  -H 'Content-Type: application/json' \
  -d '{"notes":"Confirmed stale, send nudge","actor":"rahul"}')
echo "$RES" | grep -q '"approved"' && pass "POST /alerts/$ID/approve → approved" || fail "Approve → $RES"

banner "Double-approve should fail (400)"
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/alerts/$ID/approve" \
  -H 'Content-Type: application/json' -d '{}')
[ "$CODE" = "400" ] && pass "Double-approve rejected (400)" || fail "Expected 400 on double-approve, got $CODE"

banner "Mark notified"
RES=$(curl -s -X POST "$BASE/alerts/$ID/notified?slack_ts=1234567890.123456")
echo "$RES" | grep -q '"notified"' && pass "POST /alerts/$ID/notified → notified" || fail "Notified → $RES"
echo "$RES" | grep -q '"slack_sent": true\|"slack_sent":true' && pass "slack_sent=true" || fail "slack_sent not true → $RES"

banner "Reopen notified alert"
RES=$(curl -s -X POST "$BASE/alerts/$ID/reopen" \
  -H 'Content-Type: application/json' \
  -d '{"notes":"Still blocked after nudge","actor":"rahul"}')
echo "$RES" | grep -q '"pending"' && pass "POST /alerts/$ID/reopen → pending" || fail "Reopen → $RES"

banner "Dismiss alert"
RES=$(curl -s -X POST "$BASE/alerts/$ID/dismiss" \
  -H 'Content-Type: application/json' \
  -d '{"notes":"False positive","actor":"rahul"}')
echo "$RES" | grep -q '"dismissed"' && pass "POST /alerts/$ID/dismiss → dismissed" || fail "Dismiss → $RES"

banner "Dismiss→approve should fail (400)"
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/alerts/$ID/approve" \
  -H 'Content-Type: application/json' -d '{}')
[ "$CODE" = "400" ] && pass "Dismissed→approve rejected (400)" || fail "Expected 400, got $CODE"

banner "Bulk approve"
BULK_BODY="{\"ids\": [$ID, $ID2], \"actor\": \"rahul\"}"
RES=$(curl -s -X POST "$BASE/alerts/bulk/approve" \
  -H 'Content-Type: application/json' -d "$BULK_BODY")
echo "$RES" | grep -q '"succeeded"' && pass "POST /alerts/bulk/approve" || fail "Bulk approve → $RES"

banner "Audit history"
RES=$(curl -s "$BASE/alerts/$ID/history")
# Should have at least 2 entries (creation + approve)
COUNT=$(echo "$RES" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null)
[ "$COUNT" -ge 2 ] 2>/dev/null && pass "GET /alerts/$ID/history → $COUNT entries" || fail "Expected >= 2 audit entries, got $COUNT"

banner "404 on missing alert"
CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/alerts/999999")
[ "$CODE" = "404" ] && pass "GET /alerts/999999 → 404" || fail "Expected 404, got $CODE"

echo
echo "═══════════════════════════════════════"
echo "  Results: $OK passed  │  $FAIL failed  "
echo "═══════════════════════════════════════"
[ $FAIL -eq 0 ] && exit 0 || exit 1
