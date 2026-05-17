"""Tests for session management.

Unit tests for SessionInfo and SessionStore.
"""

from datetime import datetime
from pathlib import Path

import pytest

from pentestgpt.core.session import SessionInfo, SessionStatus, SessionStore


@pytest.mark.unit
class TestSessionInfo:
    """Tests for SessionInfo dataclass."""

    def test_create_session_info(self):
        """Test creating a SessionInfo instance."""
        session = SessionInfo(
            session_id="test123",
            target="10.10.11.234",
            created_at=datetime.now(),
        )
        assert session.session_id == "test123"
        assert session.target == "10.10.11.234"
        assert session.status == SessionStatus.RUNNING
        assert session.flags_found == []
        assert session.total_cost_usd == 0.0

    def test_session_to_dict(self):
        """Test serializing SessionInfo to dict."""
        now = datetime.now()
        session = SessionInfo(
            session_id="test123",
            target="10.10.11.234",
            created_at=now,
            status=SessionStatus.COMPLETED,
            task="Test task",
            model="claude-sonnet",
        )
        data = session.to_dict()
        assert data["session_id"] == "test123"
        assert data["target"] == "10.10.11.234"
        assert data["status"] == "completed"
        assert data["task"] == "Test task"
        assert data["model"] == "claude-sonnet"

    def test_session_from_dict(self):
        """Test deserializing SessionInfo from dict."""
        data = {
            "session_id": "test456",
            "target": "example.com",
            "created_at": "2024-01-01T12:00:00",
            "status": "paused",
            "task": "Solve CTF",
            "flags_found": [{"flag": "flag{test}", "context": "Found it"}],
            "total_cost_usd": 1.5,
            "model": "claude-opus",
        }
        session = SessionInfo.from_dict(data)
        assert session.session_id == "test456"
        assert session.target == "example.com"
        assert session.status == SessionStatus.PAUSED
        assert len(session.flags_found) == 1
        assert session.total_cost_usd == 1.5


@pytest.mark.unit
class TestSessionStore:
    """Tests for SessionStore."""

    @pytest.fixture
    def session_store(self, temp_sessions_dir: Path) -> SessionStore:
        """Create a SessionStore with temp directory."""
        return SessionStore(sessions_dir=temp_sessions_dir)

    def test_create_session(self, session_store: SessionStore):
        """Test creating a new session."""
        session = session_store.create(
            target="10.10.11.234",
            task="Solve CTF",
            model="claude-sonnet",
        )
        assert session.session_id is not None
        assert len(session.session_id) == 8
        assert session.target == "10.10.11.234"
        assert session.task == "Solve CTF"
        assert session_store.current == session

    def test_save_and_load_session(self, session_store: SessionStore, temp_sessions_dir: Path):
        """Test saving and loading a session."""
        session = session_store.create(
            target="example.com",
            task="Test task",
            model="claude-opus",
        )
        session_id = session.session_id

        # Verify file was created
        session_file = temp_sessions_dir / f"{session_id}.json"
        assert session_file.exists()

        # Load the session
        loaded = session_store.load(session_id)
        assert loaded is not None
        assert loaded.session_id == session_id
        assert loaded.target == "example.com"

    def test_list_sessions(self, session_store: SessionStore):
        """Test listing sessions."""
        # Create multiple sessions
        session_store.create("target1.com", "Task 1", "model1")
        session_store.create("target2.com", "Task 2", "model2")
        session_store.create("target1.com", "Task 3", "model3")

        # List all
        all_sessions = session_store.list_sessions()
        assert len(all_sessions) == 3

        # Filter by target
        target1_sessions = session_store.list_sessions("target1.com")
        assert len(target1_sessions) == 2

    def test_get_latest_session(self, session_store: SessionStore):
        """Test getting the most recent session."""
        session_store.create("example.com", "Task 1", "model1")
        session2 = session_store.create("example.com", "Task 2", "model2")

        latest = session_store.get_latest("example.com")
        assert latest is not None
        assert latest.session_id == session2.session_id

    def test_delete_session(self, session_store: SessionStore, temp_sessions_dir: Path):
        """Test deleting a session."""
        session = session_store.create("test.com", "Test", "model")
        session_id = session.session_id

        assert session_store.delete(session_id) is True
        assert not (temp_sessions_dir / f"{session_id}.json").exists()
        assert session_store.current is None

    def test_update_session_status(self, session_store: SessionStore):
        """Test updating session status."""
        session_store.create("test.com", "Test", "model")
        session_store.update_status(SessionStatus.COMPLETED)

        assert session_store.current is not None
        assert session_store.current.status == SessionStatus.COMPLETED

    def test_add_flag(self, session_store: SessionStore):
        """Test adding a flag to session."""
        session_store.create("test.com", "Test", "model")
        session_store.add_flag("flag{test123}", "Found in output")

        assert session_store.current is not None
        assert len(session_store.current.flags_found) == 1
        assert session_store.current.flags_found[0]["flag"] == "flag{test123}"

    def test_add_cost(self, session_store: SessionStore):
        """Test adding cost to session."""
        session_store.create("test.com", "Test", "model")
        session_store.add_cost(0.5)
        session_store.add_cost(0.3)

        assert session_store.current is not None
        assert session_store.current.total_cost_usd == pytest.approx(0.8)

    def test_load_nonexistent_session(self, session_store: SessionStore):
        """Test loading a session that doesn't exist."""
        result = session_store.load("nonexistent")
        assert result is None
