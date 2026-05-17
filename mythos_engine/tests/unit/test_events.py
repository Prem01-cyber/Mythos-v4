"""Tests for event bus.

Unit tests for the EventBus singleton pattern and event emission.
"""

import pytest

from pentestgpt.core.events import Event, EventBus, EventType


@pytest.mark.unit
class TestEventBus:
    """Tests for EventBus."""

    def test_singleton(self):
        """Test EventBus singleton pattern."""
        bus1 = EventBus.get()
        bus2 = EventBus.get()
        assert bus1 is bus2

    def test_subscribe_and_emit(self):
        """Test subscribing to events and emitting."""
        bus = EventBus.get()
        received_events: list[Event] = []

        def handler(event: Event) -> None:
            received_events.append(event)

        bus.subscribe(EventType.MESSAGE, handler)
        bus.emit(Event(EventType.MESSAGE, {"text": "Hello"}))

        assert len(received_events) == 1
        assert received_events[0].data["text"] == "Hello"

    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        bus = EventBus.get()
        received_events: list[Event] = []

        def handler(event: Event) -> None:
            received_events.append(event)

        bus.subscribe(EventType.MESSAGE, handler)
        bus.unsubscribe(EventType.MESSAGE, handler)
        bus.emit(Event(EventType.MESSAGE, {"text": "Hello"}))

        assert len(received_events) == 0

    def test_multiple_handlers(self):
        """Test multiple handlers for same event type."""
        bus = EventBus.get()
        handler1_calls: list[Event] = []
        handler2_calls: list[Event] = []

        def handler1(event: Event) -> None:
            handler1_calls.append(event)

        def handler2(event: Event) -> None:
            handler2_calls.append(event)

        bus.subscribe(EventType.FLAG_FOUND, handler1)
        bus.subscribe(EventType.FLAG_FOUND, handler2)
        bus.emit_flag("flag{test}", "context")

        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1

    def test_emit_message(self):
        """Test emit_message convenience method."""
        bus = EventBus.get()
        received: list[Event] = []

        bus.subscribe(EventType.MESSAGE, lambda e: received.append(e))
        bus.emit_message("Test message", "info")

        assert len(received) == 1
        assert received[0].data["text"] == "Test message"
        assert received[0].data["type"] == "info"

    def test_emit_state(self):
        """Test emit_state convenience method."""
        bus = EventBus.get()
        received: list[Event] = []

        bus.subscribe(EventType.STATE_CHANGED, lambda e: received.append(e))
        bus.emit_state("running", "Starting agent")

        assert len(received) == 1
        assert received[0].data["state"] == "running"
        assert received[0].data["details"] == "Starting agent"

    def test_emit_tool(self):
        """Test emit_tool convenience method."""
        bus = EventBus.get()
        received: list[Event] = []

        bus.subscribe(EventType.TOOL, lambda e: received.append(e))
        bus.emit_tool("start", "bash", {"command": "ls"})

        assert len(received) == 1
        assert received[0].data["status"] == "start"
        assert received[0].data["name"] == "bash"
        assert received[0].data["args"]["command"] == "ls"

    def test_emit_flag(self):
        """Test emit_flag convenience method."""
        bus = EventBus.get()
        received: list[Event] = []

        bus.subscribe(EventType.FLAG_FOUND, lambda e: received.append(e))
        bus.emit_flag("flag{test123}", "Found in output")

        assert len(received) == 1
        assert received[0].data["flag"] == "flag{test123}"
        assert received[0].data["context"] == "Found in output"

    def test_emit_command(self):
        """Test emit_command convenience method."""
        bus = EventBus.get()
        received: list[Event] = []

        bus.subscribe(EventType.USER_COMMAND, lambda e: received.append(e))
        bus.emit_command("pause")

        assert len(received) == 1
        assert received[0].data["command"] == "pause"

    def test_emit_input(self):
        """Test emit_input convenience method."""
        bus = EventBus.get()
        received: list[Event] = []

        bus.subscribe(EventType.USER_INPUT, lambda e: received.append(e))
        bus.emit_input("Try a different approach")

        assert len(received) == 1
        assert received[0].data["text"] == "Try a different approach"

    def test_handler_exception_doesnt_break_others(self):
        """Test that one handler's exception doesn't affect others."""
        bus = EventBus.get()
        handler2_calls: list[Event] = []

        def bad_handler(event: Event) -> None:
            raise Exception("Handler error")

        def good_handler(event: Event) -> None:
            handler2_calls.append(event)

        bus.subscribe(EventType.MESSAGE, bad_handler)
        bus.subscribe(EventType.MESSAGE, good_handler)
        bus.emit_message("Test")

        # good_handler should still be called
        assert len(handler2_calls) == 1

    def test_event_has_timestamp(self):
        """Test that events have timestamps."""
        event = Event(EventType.MESSAGE, {"text": "test"})
        assert event.timestamp is not None
