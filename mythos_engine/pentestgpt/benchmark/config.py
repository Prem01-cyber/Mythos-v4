"""Simple configuration for benchmark manager."""

from pathlib import Path

# Default benchmarks directory (relative to project root)
# Use xbow-validation-benchmarks as the default benchmark suite
DEFAULT_BENCHMARKS_DIR = (
    Path(__file__).parent.parent.parent / "benchmark" / "xbow-validation-benchmarks" / "benchmarks"
)

# Default host to bind ports
DEFAULT_HOST = "0.0.0.0"

# Default port for single benchmark (can be overridden)
DEFAULT_PORT = 8080
