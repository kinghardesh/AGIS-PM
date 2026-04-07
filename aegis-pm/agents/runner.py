"""
aegis-pm / agents / runner.py

Entry point for the agent container.
APScheduler runs Monitor + Communicator on their schedules.
AgentHealthTracker records every cycle outcome and fires Slack alerts on failure.

AEGIS_MODE=agents      (default) – independent Monitor + Communicator schedules
AEGIS_MODE=groupchat   – single GroupChat full-cycle job per poll interval
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv

from agents.monitor_agent import MonitorAgent
from agents.communicator_agent import CommunicatorAgent
from agents.group_chat import run_full_cycle
from agents.health.monitor import health, AgentName

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("aegis.runner")

POLL_INTERVAL   = int(os.getenv("POLL_INTERVAL_SECONDS", "300"))
NOTIFY_INTERVAL = 30
AEGIS_MODE      = os.getenv("AEGIS_MODE", "agents").lower()


async def run_monitor():
    try:
        agent = MonitorAgent()
        loop  = asyncio.get_event_loop()

        def _run():
            with health.track(AgentName.MONITOR):
                return MonitorAgent().run_once()

        summary = await loop.run_in_executor(None, _run)
        health.record_success(
            AgentName.MONITOR,
            stale_found=summary.get("stale_found", 0),
            alerts_saved=summary.get("alerts_saved", 0),
        )
        log.info(
            "Monitor cycle – stale=%d  saved=%d  notified=%d  failed=%d",
            summary.get("stale_found", 0),
            summary.get("alerts_saved", 0),
            summary.get("notifications_sent", 0),
            summary.get("failures", 0),
        )
    except Exception as e:
        health.record_failure(AgentName.MONITOR, str(e))
        log.exception("Monitor agent crashed: %s", e)


async def run_communicator():
    try:
        loop = asyncio.get_event_loop()

        def _run():
            with health.track(AgentName.COMMUNICATOR):
                return CommunicatorAgent().run_once()

        summary = await loop.run_in_executor(None, _run)
        if summary.get("notified", 0) or summary.get("failed", 0):
            health.record_success(
                AgentName.COMMUNICATOR,
                notified=summary.get("notified", 0),
                failed=summary.get("failed", 0),
            )
            log.info(
                "Communicator cycle – approved=%d  notified=%d  failed=%d",
                summary.get("approved_found", 0),
                summary.get("notified", 0),
                summary.get("failed", 0),
            )
    except Exception as e:
        health.record_failure(AgentName.COMMUNICATOR, str(e))
        log.exception("Communicator agent crashed: %s", e)


async def run_group_chat_cycle():
    try:
        loop = asyncio.get_event_loop()

        def _run():
            with health.track(AgentName.GROUP_CHAT):
                return run_full_cycle()

        report = await loop.run_in_executor(None, _run)
        health.record_success(
            AgentName.GROUP_CHAT,
            status=report.get("status"),
            summary=report.get("summary"),
        )
        log.info(
            "GroupChat cycle – status=%s  summary=%s",
            report.get("status", "?"),
            report.get("summary", ""),
        )
    except Exception as e:
        health.record_failure(AgentName.GROUP_CHAT, str(e))
        log.exception("GroupChat cycle crashed: %s", e)


def main():
    log.info("═══════════════════════════════════════════")
    log.info("  Aegis PM – Agent Runner starting up      ")
    log.info("  Mode             : %s", AEGIS_MODE.upper())
    log.info("  Poll interval    : %ds", POLL_INTERVAL)
    if AEGIS_MODE == "agents":
        log.info("  Notify interval  : %ds", NOTIFY_INTERVAL)
    log.info("═══════════════════════════════════════════")

    scheduler = AsyncIOScheduler()

    if AEGIS_MODE == "groupchat":
        scheduler.add_job(
            run_group_chat_cycle,
            trigger=IntervalTrigger(seconds=POLL_INTERVAL),
            id="groupchat_cycle",
            name="GroupChat Full Cycle",
            next_run_time=__import__("datetime").datetime.now(),
            misfire_grace_time=120,
        )
    else:
        scheduler.add_job(
            run_monitor,
            trigger=IntervalTrigger(seconds=POLL_INTERVAL),
            id="monitor",
            name="Monitor Agent",
            next_run_time=__import__("datetime").datetime.now(),
            misfire_grace_time=60,
        )
        scheduler.add_job(
            run_communicator,
            trigger=IntervalTrigger(seconds=NOTIFY_INTERVAL),
            id="communicator",
            name="Communicator Agent",
            misfire_grace_time=30,
        )

    scheduler.start()
    loop = asyncio.get_event_loop()

    def _shutdown(sig, frame):
        log.info("Shutdown signal (%s) – stopping…", sig)
        scheduler.shutdown(wait=False)
        loop.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        scheduler.shutdown(wait=False)
        log.info("Aegis PM agent runner stopped.")


if __name__ == "__main__":
    main()
