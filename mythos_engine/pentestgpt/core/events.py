"""Event bus for decoupled communication between TUI and agent."""

import contextlib
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional


class EventType(Enum):
    """Event types for agent-TUI communication."""

    # Agent -> UI events (4 essential)
    STATE_CHANGED = auto()  # idle, running, paused, completed, error
    MESSAGE = auto()  # text output from agent
    TOOL = auto()  # tool start/complete
    FLAG_FOUND = auto()  # flag detected

    # UI -> Agent events (2 essential)
    USER_COMMAND = auto()  # pause, resume, stop
    USER_INPUT = auto()  # instruction text


@dataclass
class Event:
    """Event container with type and data."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class EventBus:
    """Minimal thread-safe event bus for pub/sub communication."""

    _instance: Optional["EventBus"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize event bus."""
        self._handlers: dict[EventType, list[Callable[[Event], None]]] = {}
        self._handler_lock = threading.Lock()

    @classmethod
    def get(cls) -> "EventBus":
        """Get singleton EventBus instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        with cls._lock:
            cls._instance = None

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Subscribe a handler to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Callback function to invoke on event
        """
        with self._handler_lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Unsubscribe a handler from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler to remove
        """
        with self._handler_lock:
            if event_type in self._handlers:
                with contextlib.suppress(ValueError):
                    self._handlers[event_type].remove(handler)

    def emit(self, event: Event) -> None:
        """Emit an event to all subscribers.

        Args:
            event: Event to emit
        """
        with self._handler_lock:
            handlers = self._handlers.get(event.type, []).copy()

        for handler in handlers:
            # Don't let one handler break others
            with contextlib.suppress(Exception):
                handler(event)

    # Convenience methods for common events

    def emit_state(
        self,
        state: str,
        details: str = "",
        target: str | None = None,
        task: str | None = None,
    ) -> None:
        """Emit a state change event.

        Args:
            state: New state (idle, running, paused, completed, error)
            details: Optional details about the state
            target: Optional target IP/URL for session tracking (used by Langfuse)
            task: Optional full task description for session tracking (used by Langfuse)
        """
        data: dict[str, Any] = {"state": state, "details": details}
        if target is not None:
            data["target"] = target
        if task is not None:
            data["task"] = task
        self.emit(Event(EventType.STATE_CHANGED, data))

    def emit_message(self, text: str, msg_type: str = "info") -> None:
        """Emit a message event.

        Args:
            text: Message text
            msg_type: Message type (info, success, error, warning)
        """
        self.emit(Event(EventType.MESSAGE, {"text": text, "type": msg_type}))

    def emit_tool(
        self,
        status: str,
        name: str,
        args: dict[str, Any] | None = None,
        result: Any | None = None,
    ) -> None:
        """Emit a tool event.

        Args:
            status: Tool status (start, complete, error)
            name: Tool name
            args: Tool arguments
            result: Tool result (for complete status)
        """
        self.emit(
            Event(
                EventType.TOOL,
                {"status": status, "name": name, "args": args or {}, "result": result},
            )
        )

    def emit_flag(self, flag: str, context: str = "") -> None:
        """Emit a flag found event.

        Args:
            flag: The flag string
            context: Context where flag was found
        """
        self.emit(Event(EventType.FLAG_FOUND, {"flag": flag, "context": context}))

    def emit_command(self, command: str) -> None:
        """Emit a user command event.

        Args:
            command: Command (pause, resume, stop)
        """
        self.emit(Event(EventType.USER_COMMAND, {"command": command}))

    def emit_input(self, text: str) -> None:
        """Emit a user input event.

        Args:
            text: User input text
        """
        self.emit(Event(EventType.USER_INPUT, {"text": text}))
