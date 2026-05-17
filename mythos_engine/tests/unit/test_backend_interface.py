"""Tests for backend interface and message types.

Unit tests for AgentBackend abstract interface and AgentMessage dataclass.
"""

import pytest

from pentestgpt.core.backend import AgentBackend, AgentMessage, MessageType


@pytest.mark.unit
class TestMessageType:
    """Tests for MessageType enum."""

    def test_message_types_exist(self):
        """Test all required message types exist."""
        assert MessageType.TEXT.value == "text"
        assert MessageType.TOOL_START.value == "tool_start"
        assert MessageType.TOOL_RESULT.value == "tool_result"
        assert MessageType.RESULT.value == "result"
        assert MessageType.ERROR.value == "error"


@pytest.mark.unit
class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_create_text_message(self):
        """Test creating a text message."""
        msg = AgentMessage(
            type=MessageType.TEXT,
            content="Hello, world!",
        )
        assert msg.type == MessageType.TEXT
        assert msg.content == "Hello, world!"
        assert msg.tool_name is None
        assert msg.tool_args is None
        assert msg.metadata == {}

    def test_create_tool_start_message(self):
        """Test creating a tool start message."""
        msg = AgentMessage(
            type=MessageType.TOOL_START,
            content=None,
            tool_name="bash",
            tool_args={"command": "ls -la"},
        )
        assert msg.type == MessageType.TOOL_START
        assert msg.tool_name == "bash"
        assert msg.tool_args is not None
        assert msg.tool_args["command"] == "ls -la"

    def test_create_tool_result_message(self):
        """Test creating a tool result message."""
        msg = AgentMessage(
            type=MessageType.TOOL_RESULT,
            content="file1.txt\nfile2.txt",
            tool_name="bash",
        )
        assert msg.type == MessageType.TOOL_RESULT
        assert msg.content == "file1.txt\nfile2.txt"

    def test_create_result_message_with_metadata(self):
        """Test creating a result message with metadata."""
        msg = AgentMessage(
            type=MessageType.RESULT,
            content=None,
            metadata={"cost_usd": 1.5, "tokens": 1000},
        )
        assert msg.type == MessageType.RESULT
        assert msg.metadata["cost_usd"] == 1.5
        assert msg.metadata["tokens"] == 1000


@pytest.mark.unit
class TestAgentBackendInterface:
    """Tests for AgentBackend abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that AgentBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AgentBackend()  # type: ignore[abstract]

    def test_interface_methods_defined(self):
        """Test that all interface methods are defined."""
        # Check abstract methods exist
        assert hasattr(AgentBackend, "connect")
        assert hasattr(AgentBackend, "disconnect")
        assert hasattr(AgentBackend, "query")
        assert hasattr(AgentBackend, "receive_messages")
        assert hasattr(AgentBackend, "session_id")
        assert hasattr(AgentBackend, "supports_resume")
        assert hasattr(AgentBackend, "resume")


class ConcreteBackend(AgentBackend):
    """Concrete implementation for testing."""

    def __init__(self) -> None:
        self._session_id = "test-session"

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def query(self, prompt: str) -> None:
        pass

    async def receive_messages(self):
        yield AgentMessage(type=MessageType.TEXT, content="test")

    @property
    def session_id(self) -> str:
        return self._session_id

    async def resume(self, session_id: str) -> bool:
        return True


@pytest.mark.unit
class TestConcreteBackend:
    """Tests for a concrete backend implementation."""

    @pytest.fixture
    def backend(self) -> ConcreteBackend:
        return ConcreteBackend()

    def test_session_id(self, backend: ConcreteBackend):
        """Test session_id property."""
        assert backend.session_id == "test-session"

    def test_supports_resume_default(self, backend: ConcreteBackend):
        """Test supports_resume default value."""
        assert backend.supports_resume is False

    @pytest.mark.asyncio
    async def test_receive_messages(self, backend: ConcreteBackend):
        """Test receive_messages iteration."""
        messages = []
        async for msg in backend.receive_messages():
            messages.append(msg)
        assert len(messages) == 1
        assert messages[0].content == "test"
