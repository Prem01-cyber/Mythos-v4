"""Activity tracer for tracking agent actions and tool executions."""

import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any


class Tracer:
    """Lightweight tracer for tracking agent activity."""

    def __init__(self) -> None:
        """Initialize the tracer."""
        self._lock = threading.Lock()
        self._activities: list[dict[str, Any]] = []
        self._on_activity_callback: Callable[[dict[str, Any]], None] | None = None

    def set_activity_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Set callback function to be called when new activity is tracked."""
        self._on_activity_callback = callback

    def track_message(
        self,
        message: str,
        message_type: str = "info",
        timestamp: datetime | None = None,
    ) -> None:
        """Track a simple message."""
        if timestamp is None:
            timestamp = datetime.now()

        activity = {
            "type": "message",
            "message": message,
            "message_type": message_type,
            "timestamp": timestamp,
        }

        with self._lock:
            self._activities.append(activity)

        if self._on_activity_callback:
            self._on_activity_callback(activity)

    def track_tool_start(
        self,
        tool_name: str,
        args: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> int:
        """Track the start of a tool execution. Returns activity ID."""
        if timestamp is None:
            timestamp = datetime.now()

        activity = {
            "type": "tool",
            "tool_name": tool_name,
            "args": args,
            "status": "running",
            "result": None,
            "timestamp": timestamp,
        }

        with self._lock:
            activity_id = len(self._activities)
            self._activities.append(activity)

        if self._on_activity_callback:
            self._on_activity_callback(activity)

        return activity_id

    def track_tool_complete(
        self,
        activity_id: int,
        result: Any = None,
        status: str = "completed",
    ) -> None:
        """Mark a tool execution as complete."""
        with self._lock:
            if 0 <= activity_id < len(self._activities):
                activity = self._activities[activity_id]
                activity["status"] = status
                activity["result"] = result

                if self._on_activity_callback:
                    self._on_activity_callback(activity)

    def track_agent_status(
        self,
        status: str,
        details: str | None = None,
    ) -> None:
        """Track agent status change."""
        message = f"Agent status: {status}"
        if details:
            message = f"{message} - {details}"

        self.track_message(message, message_type="info")

    def get_recent_activities(self, count: int = 50) -> list[dict[str, Any]]:
        """Get the most recent activities."""
        with self._lock:
            return self._activities[-count:] if self._activities else []

    def get_all_activities(self) -> list[dict[str, Any]]:
        """Get all tracked activities."""
        with self._lock:
            return self._activities.copy()

    def clear(self) -> None:
        """Clear all tracked activities."""
        with self._lock:
            self._activities.clear()


# Global tracer instance
_global_tracer: Tracer | None = None
_tracer_lock = threading.Lock()


def get_global_tracer() -> Tracer:
    """Get or create the global tracer instance."""
    global _global_tracer

    if _global_tracer is None:
        with _tracer_lock:
            if _global_tracer is None:
                _global_tracer = Tracer()

    return _global_tracer


def set_global_tracer(tracer: Tracer) -> None:
    """Set a custom global tracer."""
    global _global_tracer
    _global_tracer = tracer
