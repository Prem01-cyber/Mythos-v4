"""Tests for AgentController.

Integration tests for the agent controller lifecycle management.
"""

from pathlib import Path

import pytest

from pentestgpt.core.backend import AgentMessage, MessageType
from pentestgpt.core.config import PentestGPTConfig
from pentestgpt.core.controller import AgentController, AgentState
from pentestgpt.core.session import SessionStore
from tests.conftest import MockBackend


@pytest.mark.integration
class TestAgentController:
    """Tests for AgentController."""

    @pytest.fixture
    def config(self, temp_working_dir: Path) -> PentestGPTConfig:
        """Create test config."""
        return PentestGPTConfig(
            target="test.example.com",
            working_directory=temp_working_dir,
        )

    @pytest.fixture
    def controller(
        self,
        config: PentestGPTConfig,
        mock_backend: MockBackend,
        temp_sessions_dir: Path,
    ) -> AgentController:
        """Create controller with mock backend."""
        session_store = SessionStore(sessions_dir=temp_sessions_dir)
        return AgentController(
            config=config,
            backend=mock_backend,
            session_store=session_store,
        )

    def test_initial_state(self, controller: AgentController):
        """Test controller starts in idle state."""
        assert controller.state == AgentState.IDLE

    @pytest.mark.asyncio
    async def test_run_success(self, controller: AgentController, mock_backend: MockBackend):
        """Test successful run."""
        # Set up messages
        mock_backend.set_messages(
            [
                AgentMessage(type=MessageType.TEXT, content="Starting task..."),
                AgentMessage(type=MessageType.TEXT, content="Found flag{test123}!"),
                AgentMessage(type=MessageType.RESULT, content=None, metadata={"cost_usd": 0.5}),
            ]
        )

        result = await controller.run("Test task")

        assert result["success"] is True
        assert "flag{test123}" in result["flags_found"]
        assert result["session_id"] is not None

    @pytest.mark.asyncio
    async def test_run_with_tool_messages(
        self, controller: AgentController, mock_backend: MockBackend
    ):
        """Test run with tool messages."""
        mock_backend.set_messages(
            [
                AgentMessage(
                    type=MessageType.TOOL_START,
                    content=None,
                    tool_name="bash",
                    tool_args={"command": "nmap -sV target"},
                ),
                AgentMessage(
                    type=MessageType.TOOL_RESULT,
                    content="PORT STATE SERVICE\n22 open ssh",
                    tool_name="bash",
                ),
                AgentMessage(type=MessageType.RESULT, content=None, metadata={}),
            ]
        )

        result = await controller.run("Scan target")
        assert result["success"] is True

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skip(reason="Async pause/resume test requires concurrent task management")
    async def test_pause_and_resume(self, controller: AgentController, mock_backend: MockBackend):
        """Test pause and resume functionality.

        Note: This test is skipped because properly testing async pause/resume
        requires running the controller in a separate task and sending control
        signals concurrently. The current mock approach cannot reliably test this.
        """
        pass

    def test_pause_request(self, controller: AgentController):
        """Test pause request in wrong state."""
        # Can't pause when idle
        assert controller.pause() is False

    def test_resume_request(self, controller: AgentController):
        """Test resume request in wrong state."""
        # Can't resume when idle
        assert controller.resume() is False

    def test_stop_request(self, controller: AgentController):
        """Test stop request."""
        assert controller.stop() is True

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skip(reason="Async stop test requires concurrent task management")
    async def test_stop_during_run(self, controller: AgentController, mock_backend: MockBackend):
        """Test stopping during run.

        Note: This test is skipped because properly testing async stop
        requires running the controller in a separate task and sending stop
        signals concurrently. The current mock approach cannot reliably test this.
        """
        pass

    def test_inject_instruction(self, controller: AgentController):
        """Test inject instruction."""
        # Can't inject when idle
        assert controller.inject("New instruction") is False

    @pytest.mark.asyncio
    async def test_flag_detection(self, controller: AgentController, mock_backend: MockBackend):
        """Test flag detection patterns."""
        mock_backend.set_messages(
            [
                AgentMessage(type=MessageType.TEXT, content="Found flag{abc123}"),
                AgentMessage(type=MessageType.TEXT, content="Also HTB{test_flag}"),
                AgentMessage(
                    type=MessageType.TEXT,
                    content="And 0123456789abcdef0123456789abcdef",
                ),
                AgentMessage(type=MessageType.RESULT, content=None, metadata={}),
            ]
        )

        result = await controller.run("Find flags")

        assert len(result["flags_found"]) == 3
        assert "flag{abc123}" in result["flags_found"]
        assert "HTB{test_flag}" in result["flags_found"]

    @pytest.mark.asyncio
    async def test_session_not_found(self, controller: AgentController):
        """Test resuming non-existent session."""
        result = await controller.run("Task", resume_session_id="nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_backend_error(self, controller: AgentController, mock_backend: MockBackend):
        """Test handling backend errors."""

        async def error_messages():
            raise Exception("Backend error")
            yield  # Make it a generator

        mock_backend.receive_messages = error_messages  # type: ignore[method-assign]

        result = await controller.run("Test task")
        assert result["success"] is False
        assert "Backend error" in result["error"]
