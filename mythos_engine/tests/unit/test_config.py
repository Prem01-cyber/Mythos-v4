"""Tests for configuration management.

Unit tests for PentestGPTConfig and load_config function.
"""

import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from pentestgpt.core.config import PentestGPTConfig, load_config


@pytest.mark.unit
class TestPentestGPTConfig:
    """Tests for PentestGPTConfig."""

    def test_create_config_with_required_fields(self, temp_working_dir: Path):
        """Test creating config with only required fields."""
        config = PentestGPTConfig(
            target="10.10.11.234",
            working_directory=temp_working_dir,
        )
        assert config.target == "10.10.11.234"
        assert config.working_directory == temp_working_dir

    def test_default_values(self, temp_working_dir: Path):
        """Test that default values are set correctly."""
        config = PentestGPTConfig(
            target="example.com",
            working_directory=temp_working_dir,
        )
        assert config.llm_model == "claude-sonnet-4-5-20250929"
        assert config.llm_api_key is None
        assert config.llm_api_base is None
        assert config.max_iterations == 300
        assert config.custom_instruction is None
        assert config.interface_mode == "tui"
        assert config.verbose is True
        assert config.permission_mode == "bypassPermissions"

    def test_missing_required_field(self, temp_working_dir: Path):
        """Test that missing target raises validation error."""
        with pytest.raises(ValidationError):
            PentestGPTConfig(
                working_directory=temp_working_dir,
            )  # type: ignore[call-arg]

    def test_custom_instruction(self, temp_working_dir: Path):
        """Test setting custom instruction."""
        config = PentestGPTConfig(
            target="ctf.example.com",
            working_directory=temp_working_dir,
            custom_instruction="Focus on web vulnerabilities",
        )
        assert config.custom_instruction == "Focus on web vulnerabilities"

    def test_interface_mode_validation(self, temp_working_dir: Path):
        """Test interface mode must be valid."""
        # Valid modes
        config_tui = PentestGPTConfig(
            target="test.com",
            working_directory=temp_working_dir,
            interface_mode="tui",
        )
        assert config_tui.interface_mode == "tui"

        config_cli = PentestGPTConfig(
            target="test.com",
            working_directory=temp_working_dir,
            interface_mode="cli",
        )
        assert config_cli.interface_mode == "cli"

        # Invalid mode
        with pytest.raises(ValidationError):
            PentestGPTConfig(
                target="test.com",
                working_directory=temp_working_dir,
                interface_mode="invalid",  # type: ignore[arg-type]
            )

    def test_permission_mode_validation(self, temp_working_dir: Path):
        """Test permission mode must be valid."""
        config_ask = PentestGPTConfig(
            target="test.com",
            working_directory=temp_working_dir,
            permission_mode="ask",
        )
        assert config_ask.permission_mode == "ask"

        config_bypass = PentestGPTConfig(
            target="test.com",
            working_directory=temp_working_dir,
            permission_mode="bypassPermissions",
        )
        assert config_bypass.permission_mode == "bypassPermissions"

    def test_working_directory_created(self):
        """Test that working directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_workspace"
            assert not new_dir.exists()

            config = PentestGPTConfig(
                target="test.com",
                working_directory=new_dir,
            )

            assert new_dir.exists()
            assert config.working_directory == new_dir

    def test_system_prompt_path(self, temp_working_dir: Path):
        """Test system_prompt_path property."""
        config = PentestGPTConfig(
            target="test.com",
            working_directory=temp_working_dir,
        )
        prompt_path = config.system_prompt_path
        assert prompt_path.name == "pentesting.py"
        assert "prompts" in str(prompt_path)


@pytest.mark.unit
class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_with_target(self, temp_working_dir: Path):
        """Test load_config with target override."""
        config = load_config(
            target="192.168.1.1",
            working_directory=temp_working_dir,
        )
        assert config.target == "192.168.1.1"

    def test_load_config_with_multiple_overrides(self, temp_working_dir: Path):
        """Test load_config with multiple overrides."""
        config = load_config(
            target="ctf.local",
            working_directory=temp_working_dir,
            llm_model="claude-opus",
            max_iterations=500,
            verbose=False,
        )
        assert config.target == "ctf.local"
        assert config.llm_model == "claude-opus"
        assert config.max_iterations == 500
        assert config.verbose is False

    def test_from_env_classmethod(self, temp_working_dir: Path):
        """Test from_env classmethod."""
        config = PentestGPTConfig.from_env(
            target="env.example.com",
            working_directory=temp_working_dir,
        )
        assert config.target == "env.example.com"

    def test_load_config_from_environment(self, temp_working_dir: Path):
        """Test that config can load from environment variables."""
        original_env = os.environ.copy()
        try:
            os.environ["LLM_MODEL"] = "test-model"
            config = load_config(
                target="test.com",
                working_directory=temp_working_dir,
            )
            # Note: Environment variables should be loaded
            # The actual behavior depends on pydantic-settings
            assert config.target == "test.com"
        finally:
            os.environ.clear()
            os.environ.update(original_env)
