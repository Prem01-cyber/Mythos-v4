"""Tests for benchmark registry.

Unit tests for BenchmarkInfo and BenchmarkRegistry.
"""

import json
import tempfile
from pathlib import Path

import pytest

from pentestgpt.benchmark.registry import BenchmarkRegistry


@pytest.mark.unit
class TestBenchmarkRegistry:
    """Test benchmark registry functionality."""

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
                        "name": "Test Benchmark 1",
                        "description": "Test description",
                        "level": "2",
                        "tags": ["sqli", "xss"],
                    }
                )
            )

            # Create benchmark 2
            bench2 = benchmarks_dir / "XBEN-002-24"
            bench2.mkdir()
            (bench2 / "benchmark.json").write_text(
                json.dumps(
                    {
                        "name": "Test Benchmark 2",
                        "description": "Another test",
                        "level": 1,
                        "tags": ["lfi"],
                    }
                )
            )

            yield benchmarks_dir

    def test_load_benchmarks(self, sample_benchmark_dir: Path) -> None:
        """Test loading benchmarks."""
        registry = BenchmarkRegistry(sample_benchmark_dir)
        registry.load()
        assert len(registry.list_all()) == 2

    def test_get_benchmark(self, sample_benchmark_dir: Path) -> None:
        """Test getting benchmark by ID."""
        registry = BenchmarkRegistry(sample_benchmark_dir)
        benchmark = registry.get("XBEN-001-24")
        assert benchmark is not None
        assert benchmark.name == "Test Benchmark 1"

    def test_filter_by_tags(self, sample_benchmark_dir: Path) -> None:
        """Test filtering by tags."""
        registry = BenchmarkRegistry(sample_benchmark_dir)
        results = registry.filter(tags=["sqli"])
        assert len(results) == 1
        assert results[0].id == "XBEN-001-24"

    def test_filter_by_levels(self, sample_benchmark_dir: Path) -> None:
        """Test filtering by levels."""
        registry = BenchmarkRegistry(sample_benchmark_dir)
        results = registry.filter(levels=[1])
        assert len(results) == 1
        assert results[0].id == "XBEN-002-24"

    def test_get_all_tags(self, sample_benchmark_dir: Path) -> None:
        """Test getting all tags."""
        registry = BenchmarkRegistry(sample_benchmark_dir)
        tags = registry.get_all_tags()
        assert tags == {"sqli", "xss", "lfi"}

    def test_nonexistent_benchmark(self, sample_benchmark_dir: Path) -> None:
        """Test getting nonexistent benchmark."""
        registry = BenchmarkRegistry(sample_benchmark_dir)
        assert registry.get("XBEN-999-24") is None
