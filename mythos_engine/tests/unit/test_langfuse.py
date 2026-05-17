"""Tests for Langfuse observability integration.

Unit tests for the Langfuse event handler module (SDK v3 API).
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pentestgpt.core.events import Event, EventBus, EventType


@pytest.mark.unit
class TestLangfuseIntegration:
    """Tests for Langfuse integration."""

    def test_init_langfuse_disabled_via_flag(self):
        """Test that Langfuse is not initialized when disabled flag is True."""
        import importlib

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        result = langfuse_module.init_langfuse(disabled=True)
        assert result is False

    def test_init_langfuse_disabled_via_env_var(self):
        """Test that Langfuse is not initialized when LANGFUSE_ENABLED=false."""
        env_backup = os.environ.get("LANGFUSE_ENABLED")
        os.environ["LANGFUSE_ENABLED"] = "false"

        try:
            import importlib

            import pentestgpt.core.langfuse as langfuse_module

            importlib.reload(langfuse_module)

            result = langfuse_module.init_langfuse()
            assert result is False
        finally:
            if env_backup is not None:
                os.environ["LANGFUSE_ENABLED"] = env_backup
            else:
                os.environ.pop("LANGFUSE_ENABLED", None)

    def test_init_langfuse_with_mock_v3_api(self):
        """Test Langfuse initialization uses v3 get_client() API."""
        env_backup = os.environ.get("LANGFUSE_ENABLED")

        try:
            os.environ.pop("LANGFUSE_ENABLED", None)

            import importlib
            import sys

            import pentestgpt.core.langfuse as langfuse_module

            importlib.reload(langfuse_module)

            mock_langfuse_module = MagicMock()
            mock_client = MagicMock()
            mock_langfuse_module.get_client.return_value = mock_client

            with patch.dict(sys.modules, {"langfuse": mock_langfuse_module}):
                result = langfuse_module.init_langfuse()

            # Should use v3 get_client() API
            mock_langfuse_module.get_client.assert_called_once()
            assert result is True
            # Should have a user ID set
            assert langfuse_module._user_id is not None
        finally:
            if env_backup is not None:
                os.environ["LANGFUSE_ENABLED"] = env_backup
            else:
                os.environ.pop("LANGFUSE_ENABLED", None)

    def test_get_or_create_user_id_creates_new(self):
        """Test that a new user ID is created if none exists."""
        import importlib

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(Path, "home", return_value=Path(tmpdir)),
        ):
            user_id = langfuse_module._get_or_create_user_id()

            # Should be a valid UUID format
            assert len(user_id) == 36
            assert user_id.count("-") == 4

            # File should exist
            user_id_file = Path(tmpdir) / ".pentestgpt" / "user_id"
            assert user_id_file.exists()
            assert user_id_file.read_text().strip() == user_id

    def test_get_or_create_user_id_reads_existing(self):
        """Test that existing user ID is read from file."""
        import importlib

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing user ID file
            pentestgpt_dir = Path(tmpdir) / ".pentestgpt"
            pentestgpt_dir.mkdir(parents=True)
            user_id_file = pentestgpt_dir / "user_id"
            existing_id = "existing-test-uuid-1234"
            user_id_file.write_text(existing_id)

            # Read the existing ID
            with patch.object(Path, "home", return_value=Path(tmpdir)):
                user_id = langfuse_module._get_or_create_user_id()
                assert user_id == existing_id

    def test_shutdown_langfuse_safe_when_not_initialized(self):
        """Test that shutdown_langfuse is safe to call when not initialized."""
        import importlib

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        # Should not raise any exception
        langfuse_module.shutdown_langfuse()

    def test_event_handlers_guard_against_no_client(self):
        """Test that event handlers safely do nothing when client is None."""
        import importlib

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        # Ensure client is None
        langfuse_module._langfuse_client = None
        langfuse_module._current_span = None

        # These should not raise exceptions
        langfuse_module._handle_state(Event(EventType.STATE_CHANGED, {"state": "running"}))
        langfuse_module._handle_message(Event(EventType.MESSAGE, {"text": "test"}))
        langfuse_module._handle_tool(Event(EventType.TOOL, {"status": "start", "name": "bash"}))
        langfuse_module._handle_flag(Event(EventType.FLAG_FOUND, {"flag": "test"}))

    def test_state_handler_stores_pending_session_on_running(self):
        """Test that state handler stores pending session data on 'running' state (deferred creation)."""
        import importlib

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        mock_client = MagicMock()
        langfuse_module._langfuse_client = mock_client
        langfuse_module._current_span = None
        langfuse_module._user_id = "test-user-id-1234"

        langfuse_module._handle_state(
            Event(
                EventType.STATE_CHANGED,
                {"state": "running", "target": "10.10.11.234", "task": "Solve CTF"},
            )
        )

        # Span should NOT be created yet (deferred until first tool)
        mock_client.start_span.assert_not_called()
        assert langfuse_module._current_span is None

        # Pending session should be stored
        assert langfuse_module._pending_session is not None
        assert langfuse_module._pending_session["target"] == "10.10.11.234"
        assert langfuse_module._pending_session["task"] == "Solve CTF"
        assert "session_id" in langfuse_module._pending_session

    def test_state_handler_uses_target_field(self):
        """Test that state handler uses the new target field when provided."""
        import importlib

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        mock_client = MagicMock()
        langfuse_module._langfuse_client = mock_client
        langfuse_module._current_span = None
        langfuse_module._user_id = "test-user-id-1234"

        langfuse_module._handle_state(
            Event(
                EventType.STATE_CHANGED,
                {
                    "state": "running",
                    "details": "Connecting...",
                    "target": "10.10.11.234",
                    "task": "My task",
                },
            )
        )

        # Should use target field, not details
        assert langfuse_module._session_target == "10.10.11.234"
        assert langfuse_module._pending_session["target"] == "10.10.11.234"
        assert langfuse_module._pending_session["task"] == "My task"

    def test_state_handler_backward_compatible_with_details_only(self):
        """Test that state handler falls back to details if target not provided."""
        import importlib

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        mock_client = MagicMock()
        langfuse_module._langfuse_client = mock_client
        langfuse_module._current_span = None
        langfuse_module._user_id = "test-user-id-1234"

        # Old-style event without target field
        langfuse_module._handle_state(
            Event(EventType.STATE_CHANGED, {"state": "running", "details": "10.10.11.234"})
        )

        # Should fall back to details as target
        assert langfuse_module._session_target == "10.10.11.234"
        assert langfuse_module._pending_session["target"] == "10.10.11.234"

    def test_state_handler_ends_span_on_completed(self):
        """Test that state handler ends span and flushes on completion (v3 API)."""
        import importlib

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        mock_client = MagicMock()
        mock_span = MagicMock()

        langfuse_module._langfuse_client = mock_client
        langfuse_module._current_span = mock_span
        langfuse_module._session_target = "10.10.11.234"

        langfuse_module._handle_state(Event(EventType.STATE_CHANGED, {"state": "completed"}))

        mock_span.update.assert_called_once_with(
            output={"final_state": "completed", "target": "10.10.11.234"}
        )
        mock_span.end.assert_called_once()
        mock_client.flush.assert_called_once()
        assert langfuse_module._current_span is None
        assert langfuse_module._session_target is None

    def test_message_handler_creates_nested_span(self):
        """Test that message handler creates a nested span (v3 API)."""
        import importlib

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        mock_client = MagicMock()
        mock_span = MagicMock()
        mock_nested_span = MagicMock()
        mock_span.start_span.return_value = mock_nested_span

        langfuse_module._langfuse_client = mock_client
        langfuse_module._current_span = mock_span

        langfuse_module._handle_message(
            Event(EventType.MESSAGE, {"text": "Hello world", "type": "info"})
        )

        mock_span.start_span.assert_called_once_with(
            name="agent-message",
            input={"message_type": "info"},
            output={"text": "Hello world"},
        )
        mock_nested_span.end.assert_called_once()

    def test_tool_handler_creates_session_span_on_first_tool(self):
        """Test that tool handler creates session span on first tool (deferred creation)."""
        import importlib

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        mock_client = MagicMock()
        mock_session_span = MagicMock()
        mock_tool_span = MagicMock()
        mock_client.start_span.return_value = mock_session_span
        mock_session_span.start_span.return_value = mock_tool_span

        langfuse_module._langfuse_client = mock_client
        langfuse_module._current_span = None
        langfuse_module._user_id = "test-user-id-1234"
        langfuse_module._pending_session = {
            "target": "10.10.11.234",
            "task": "Solve CTF",
            "session_id": "test-1234-session",
        }

        langfuse_module._handle_tool(
            Event(EventType.TOOL, {"status": "start", "name": "bash", "args": {"cmd": "ls"}})
        )

        # Session span should now be created
        mock_client.start_span.assert_called_once()
        call_kwargs = mock_client.start_span.call_args.kwargs
        assert call_kwargs["name"] == "pentestgpt:10.10.11.234"
        assert call_kwargs["input"]["target"] == "10.10.11.234"
        assert call_kwargs["input"]["task"] == "Solve CTF"
        assert call_kwargs["metadata"]["task"] == "Solve CTF"

        # Pending session should be cleared
        assert langfuse_module._pending_session is None
        assert langfuse_module._tool_executed is True
        assert langfuse_module._current_span is mock_session_span

        # Tool span should be created as nested span
        mock_session_span.start_span.assert_called_once()
        mock_tool_span.end.assert_called_once()

    def test_tool_handler_creates_nested_span_when_span_exists(self):
        """Test that tool handler creates nested span when session span already exists."""
        import importlib

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        mock_client = MagicMock()
        mock_span = MagicMock()
        mock_nested_span = MagicMock()
        mock_span.start_span.return_value = mock_nested_span

        langfuse_module._langfuse_client = mock_client
        langfuse_module._current_span = mock_span
        langfuse_module._pending_session = None  # Already processed

        langfuse_module._handle_tool(
            Event(EventType.TOOL, {"status": "start", "name": "bash", "args": {"cmd": "ls"}})
        )

        mock_span.start_span.assert_called_once_with(
            name="tool-bash",
            input={"cmd": "ls"},
            metadata={"tool_name": "bash"},
        )
        mock_nested_span.end.assert_called_once()

    def test_session_discarded_if_no_tools_executed(self):
        """Test that session is discarded if completed without any tool execution."""
        import importlib

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        mock_client = MagicMock()
        langfuse_module._langfuse_client = mock_client
        langfuse_module._current_span = None  # No span created
        langfuse_module._session_target = "10.10.11.234"
        langfuse_module._pending_session = {
            "target": "10.10.11.234",
            "task": "Solve CTF",
            "session_id": "test-session",
        }

        # Complete session without any tools
        langfuse_module._handle_state(Event(EventType.STATE_CHANGED, {"state": "completed"}))

        # No span should have been created
        mock_client.start_span.assert_not_called()
        mock_client.flush.assert_not_called()

        # State should be reset
        assert langfuse_module._pending_session is None
        assert langfuse_module._session_target is None
        assert langfuse_module._current_span is None

    def test_flag_handler_creates_nested_span(self):
        """Test that flag handler creates a nested span (v3 API)."""
        import importlib

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        mock_client = MagicMock()
        mock_span = MagicMock()
        mock_nested_span = MagicMock()
        mock_span.start_span.return_value = mock_nested_span

        langfuse_module._langfuse_client = mock_client
        langfuse_module._current_span = mock_span

        langfuse_module._handle_flag(
            Event(EventType.FLAG_FOUND, {"flag": "flag{test}", "context": "Found in output"})
        )

        mock_span.start_span.assert_called_once_with(
            name="flag-found",
            input={"context": "Found in output"},
            output={"flag": "flag{test}"},
            metadata={"flag": "flag{test}", "context": "Found in output"},
        )
        mock_nested_span.end.assert_called_once()

    def test_eventbus_integration(self):
        """Test that handlers are properly subscribed to EventBus (v3 API)."""
        import importlib

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        mock_client = MagicMock()
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        langfuse_module._langfuse_client = mock_client
        langfuse_module._current_span = None
        langfuse_module._user_id = "test-user-id-1234"

        # Subscribe to events
        langfuse_module._subscribe_to_events()

        # Emit running state via EventBus (should create pending session, not span)
        bus = EventBus.get()
        bus.emit_state("running", "Connecting...", target="10.10.11.234", task="Test task")

        # Span should NOT be created yet (deferred)
        mock_client.start_span.assert_not_called()
        assert langfuse_module._pending_session is not None
        assert langfuse_module._pending_session["target"] == "10.10.11.234"

        # Emit tool event - this should create the span
        bus.emit_tool("start", "bash", {"cmd": "ls"})

        # Now span should be created
        mock_client.start_span.assert_called()

    def test_langfuse_loggers_silenced(self):
        """Test that noisy loggers are silenced during init."""
        import importlib
        import logging

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        # Call the silence function
        langfuse_module._silence_langfuse_loggers()

        # Check that noisy loggers are silenced
        noisy_loggers = [
            "langfuse",
            "opentelemetry",
            "opentelemetry.sdk",
            "opentelemetry.sdk._shared_internal",
        ]

        for logger_name in noisy_loggers:
            test_logger = logging.getLogger(logger_name)
            # Logger should be at CRITICAL+1 level (effectively silenced)
            assert test_logger.level > logging.CRITICAL
            # Propagate should be False
            assert test_logger.propagate is False

    def test_shutdown_resets_all_state(self):
        """Test that shutdown resets all module state including new variables."""
        import importlib

        import pentestgpt.core.langfuse as langfuse_module

        importlib.reload(langfuse_module)

        # Set up some state
        langfuse_module._langfuse_client = MagicMock()
        langfuse_module._current_span = MagicMock()
        langfuse_module._user_id = "test-user"
        langfuse_module._session_target = "10.10.11.234"
        langfuse_module._pending_session = {"target": "test"}
        langfuse_module._tool_executed = True

        # Shutdown
        langfuse_module.shutdown_langfuse()

        # All state should be reset
        assert langfuse_module._langfuse_client is None
        assert langfuse_module._current_span is None
        assert langfuse_module._user_id is None
        assert langfuse_module._session_target is None
        assert langfuse_module._pending_session is None
        assert langfuse_module._tool_executed is False
