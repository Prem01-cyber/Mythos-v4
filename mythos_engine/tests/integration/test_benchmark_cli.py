"""Tests for benchmark CLI commands.

Integration tests for the benchmark CLI command handlers.
"""

import json
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pentestgpt.benchmark.cli import cmd_list, cmd_start, cmd_status, cmd_stop


@pytest.mark.integration
class TestBenchmarkCLI:
    """Tests for benchmark CLI commands."""

    @pytest.fixture
    def sample_benchmark_dir(self) -> Path:
        """Create sample benchmark structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmarks_dir = Path(tmpdir)

            # Create benchmark 1
            bench1 = benchmarks_dir / "XBEN-001-24"
            bench1.mkdir()
            (bench1 / "benchmark.json").write_text(
                json.dumps(
                    {
                        "name": "Test SQL Injection",
                        "description": "A simple SQL injection challenge",
                        "level": 1,
                        "tags": ["sqli", "web"],
                    }
                )
            )

            # Create benchmark 2
            bench2 = benchmarks_dir / "XBEN-002-24"
            bench2.mkdir()
            (bench2 / "benchmark.json").write_text(
                json.dumps(
                    {
                        "name": "Advanced XSS",
                        "description": "Cross-site scripting challenge",
                        "level": 2,
                        "tags": ["xss", "web"],
                    }
                )
            )

            # Create benchmark 3
            bench3 = benchmarks_dir / "XBEN-003-24"
            bench3.mkdir()
            (bench3 / "benchmark.json").write_text(
                json.dumps(
                    {
                        "name": "File Inclusion",
                        "description": "LFI/RFI challenge",
                        "level": 3,
                        "tags": ["lfi", "rfi"],
                    }
                )
            )

            yield benchmarks_dir

    def test_cmd_list_all(self, sample_benchmark_dir: Path, capsys):
        """Test listing all benchmarks."""
        args = Namespace(
            benchmarks_dir=str(sample_benchmark_dir),
            tags=None,
            levels=None,
            show_tags=False,
        )
        result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "XBEN-001-24" in captured.out
        assert "XBEN-002-24" in captured.out
        assert "XBEN-003-24" in captured.out
        assert "Total: 3 benchmarks" in captured.out

    def test_cmd_list_filter_by_tag(self, sample_benchmark_dir: Path, capsys):
        """Test listing benchmarks filtered by tag."""
        args = Namespace(
            benchmarks_dir=str(sample_benchmark_dir),
            tags=["sqli"],
            levels=None,
            show_tags=False,
        )
        result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "XBEN-001-24" in captured.out
        assert "XBEN-002-24" not in captured.out
        assert "Total: 1 benchmarks" in captured.out

    def test_cmd_list_filter_by_level(self, sample_benchmark_dir: Path, capsys):
        """Test listing benchmarks filtered by level."""
        args = Namespace(
            benchmarks_dir=str(sample_benchmark_dir),
            tags=None,
            levels=[1, 2],
            show_tags=False,
        )
        result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "XBEN-001-24" in captured.out
        assert "XBEN-002-24" in captured.out
        assert "XBEN-003-24" not in captured.out
        assert "Total: 2 benchmarks" in captured.out

    def test_cmd_list_show_tags(self, sample_benchmark_dir: Path, capsys):
        """Test showing all tags."""
        args = Namespace(
            benchmarks_dir=str(sample_benchmark_dir),
            tags=None,
            levels=None,
            show_tags=True,
        )
        result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Available tags:" in captured.out
        assert "sqli:" in captured.out
        assert "web:" in captured.out
        assert "xss:" in captured.out

    def test_cmd_list_invalid_dir(self, capsys):
        """Test listing from non-existent directory."""
        args = Namespace(
            benchmarks_dir="/nonexistent/path",
            tags=None,
            levels=None,
            show_tags=False,
        )
        result = cmd_list(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.out

    @patch("pentestgpt.benchmark.cli.start_benchmark")
    def test_cmd_start_success(self, mock_start: MagicMock, sample_benchmark_dir: Path, capsys):
        """Test starting a benchmark successfully."""
        mock_start.return_value = {
            "success": True,
            "target_url": "http://localhost:8080",
        }

        args = Namespace(
            benchmarks_dir=str(sample_benchmark_dir),
            benchmark_id="XBEN-001-24",
        )
        result = cmd_start(args)

        assert result == 0
        mock_start.assert_called_once()
        captured = capsys.readouterr()
        assert "Benchmark started successfully!" in captured.out
        assert "http://localhost:8080" in captured.out

    @patch("pentestgpt.benchmark.cli.start_benchmark")
    def test_cmd_start_failure(self, mock_start: MagicMock, sample_benchmark_dir: Path, capsys):
        """Test handling failed benchmark start."""
        mock_start.return_value = {
            "success": False,
            "message": "Docker error",
        }

        args = Namespace(
            benchmarks_dir=str(sample_benchmark_dir),
            benchmark_id="XBEN-001-24",
        )
        result = cmd_start(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Failed to start benchmark" in captured.out
        assert "Docker error" in captured.out

    def test_cmd_start_not_found(self, sample_benchmark_dir: Path, capsys):
        """Test starting non-existent benchmark."""
        args = Namespace(
            benchmarks_dir=str(sample_benchmark_dir),
            benchmark_id="XBEN-999-24",
        )
        result = cmd_start(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out

    @patch("pentestgpt.benchmark.cli.stop_benchmark")
    def test_cmd_stop_success(self, mock_stop: MagicMock, sample_benchmark_dir: Path, capsys):
        """Test stopping a benchmark successfully."""
        mock_stop.return_value = {"success": True}

        args = Namespace(
            benchmarks_dir=str(sample_benchmark_dir),
            benchmark_id="XBEN-001-24",
        )
        result = cmd_stop(args)

        assert result == 0
        mock_stop.assert_called_once()
        captured = capsys.readouterr()
        assert "stopped successfully" in captured.out

    @patch("pentestgpt.benchmark.cli.stop_benchmark")
    def test_cmd_stop_failure(self, mock_stop: MagicMock, sample_benchmark_dir: Path, capsys):
        """Test handling failed benchmark stop."""
        mock_stop.return_value = {
            "success": False,
            "message": "Container not running",
        }

        args = Namespace(
            benchmarks_dir=str(sample_benchmark_dir),
            benchmark_id="XBEN-001-24",
        )
        result = cmd_stop(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Failed to stop" in captured.out

    def test_cmd_stop_not_found(self, sample_benchmark_dir: Path, capsys):
        """Test stopping non-existent benchmark."""
        args = Namespace(
            benchmarks_dir=str(sample_benchmark_dir),
            benchmark_id="XBEN-999-24",
        )
        result = cmd_stop(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out

    @patch("pentestgpt.benchmark.cli.get_running_benchmarks")
    def test_cmd_status_running(self, mock_running: MagicMock, capsys):
        """Test showing running benchmarks."""
        mock_running.return_value = [
            {
                "name": "xben-001-24",
                "ports": "0.0.0.0:8080->80/tcp",
                "status": "Up 5 minutes",
            }
        ]

        args = Namespace()
        result = cmd_status(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Running benchmark containers:" in captured.out
        assert "xben-001-24" in captured.out
        assert "8080" in captured.out

    @patch("pentestgpt.benchmark.cli.get_running_benchmarks")
    def test_cmd_status_none_running(self, mock_running: MagicMock, capsys):
        """Test showing status when no benchmarks running."""
        mock_running.return_value = []

        args = Namespace()
        result = cmd_status(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No benchmark containers currently running" in captured.out
