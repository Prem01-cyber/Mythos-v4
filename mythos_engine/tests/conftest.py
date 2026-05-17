"""Shared pytest fixtures for PentestGPT tests."""

import tempfile
from pathlib import Path

import pytest

from pentestgpt.core.backend import AgentBackend, AgentMessage
from pentestgpt.core.config import PentestGPTConfig
from pentestgpt.core.events import EventBus

# =============================================================================
# Pytest Markers
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external dependencies)")
    config.addinivalue_line("markers", "integration: Integration tests (may use mocks)")
    config.addinivalue_line("markers", "docker: Docker tests (requires Docker daemon)")
    config.addinivalue_line("markers", "slow: Slow tests (skip with -m 'not slow')")


# =============================================================================
# EventBus Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_event_bus():
    """Reset EventBus singleton before and after each test."""
    EventBus.reset()
    yield
    EventBus.reset()


# =============================================================================
# Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_sessions_dir():
    """Create a temporary directory for session storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_working_dir():
    """Create a temporary working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def sample_config(temp_working_dir: Path) -> PentestGPTConfig:
    """Create a sample configuration for testing."""
    return PentestGPTConfig(
        target="test.example.com",
        working_directory=temp_working_dir,
    )


# =============================================================================
# Mock Backend
# =============================================================================


class MockBackend(AgentBackend):
    """Mock backend for testing agent controller."""

    def __init__(self) -> None:
        self._connected = False
        self._messages: list[AgentMessage] = []
        self._session_id = "mock-session-123"

    async def connect(self) -> None:
        """Simulate connection."""
        self._connected = True

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self._connected = False

    async def query(self, prompt: str) -> None:
        """Simulate sending a query."""
        pass

    async def receive_messages(self):
        """Yield preset messages."""
        for msg in self._messages:
            yield msg

    @property
    def session_id(self) -> str:
        """Get mock session ID."""
        return self._session_id

    @property
    def supports_resume(self) -> bool:
        """Mock does not support resume."""
        return False

    async def resume(self, session_id: str) -> bool:
        """Mock resume always fails."""
        return False

    def set_messages(self, messages: list[AgentMessage]) -> None:
        """Set messages to be returned by receive_messages."""
        self._messages = messages


@pytest.fixture
def mock_backend() -> MockBackend:
    """Create a mock backend for testing."""
    return MockBackend()
