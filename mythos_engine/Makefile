# PentestGPT Makefile
# Usage: make [target]

.PHONY: help install config connect start stop shell logs clean-docker
.PHONY: dev-install test test-cov test-verbose lint format typecheck clean build
.PHONY: ci ci-quick ci-full

# Default target
help:
	@echo "PentestGPT Commands"
	@echo "==================="
	@echo ""
	@echo "Docker Workflow (Primary Usage):"
	@echo "  make install         Install dependencies (uv sync) and build Docker image"
	@echo "  make config          Configure authentication (interactive)"
	@echo "                       Options: Claude Login, OpenRouter, Anthropic API, Local LLM"
	@echo "  make connect         Connect to container (main entry point)"
	@echo "  make start           Start container in background"
	@echo "  make stop            Stop container (keeps config)"
	@echo "  make shell           Open new shell in running container"
	@echo "  make logs            View container logs"
	@echo "  make clean-docker    Remove everything including config"
	@echo ""
	@echo "Development:"
	@echo "  make dev-install  Install dev dependencies locally"
	@echo "  make test         Run all tests"
	@echo "  make lint         Run linter (ruff)"
	@echo "  make format       Format code (ruff)"
	@echo "  make typecheck    Run type checker (mypy)"
	@echo "  make check        Run all checks (lint + typecheck)"
	@echo "  make ci           Run full CI simulation (lint, format, typecheck, test, build)"
	@echo "  make ci-quick     Run quick CI (skip build step)"
	@echo "  make ci-full      Run CI with Docker tests (requires Docker)"
	@echo "  make clean        Clean build artifacts"

# ============================================================================
# Docker Workflow (Primary Usage)
# ============================================================================

# Build the Docker image and install local dependencies
install:
	@echo "Installing local dependencies with uv..."
	uv sync
	@echo "Building PentestGPT Docker image..."
	docker compose build --no-cache

# Configure authentication (interactive menu)
config:
	@chmod +x scripts/config.sh
	@./scripts/config.sh

# Connect to the running container (main entry point)
# Handles different auth modes automatically based on .env.auth
connect:
	@if [ "$$(docker ps -q -f name=pentestgpt)" ]; then \
		echo "Attaching to running container..."; \
		docker attach pentestgpt; \
	else \
		echo "Starting new container..."; \
		if [ -f .env.auth ]; then \
			docker compose --env-file .env.auth up -d && docker attach pentestgpt; \
		else \
			docker compose up -d && docker attach pentestgpt; \
		fi; \
	fi

# Start container in background
start:
	@if [ -f .env.auth ]; then \
		docker compose --env-file .env.auth up -d; \
	else \
		docker compose up -d; \
	fi

# Stop and remove container (keeps config volume)
stop:
	docker compose down

# Execute command in running container
shell:
	docker exec -it pentestgpt /bin/bash

# View container logs
logs:
	docker compose logs -f

# Clean up everything including volumes and auth config
clean-docker:
	docker compose down -v
	docker rmi pentestgpt:latest 2>/dev/null || true
	rm -f .env.auth

# ============================================================================
# Local Development Setup
# ============================================================================

dev-install:
	uv sync

# ============================================================================
# Testing
# ============================================================================

test:
	uv run pytest tests/ -v --ignore=tests/docker/

test-all:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --ignore=tests/docker/ --cov=pentestgpt --cov-report=term-missing --cov-report=html

test-verbose:
	uv run pytest tests/ -vvs --ignore=tests/docker/

# Test by category
test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-docker:
	uv run pytest tests/docker/ -v -m docker

test-fast:
	uv run pytest tests/ -v -m "not slow and not docker" --ignore=tests/docker/

# Run specific test files
test-session:
	uv run pytest tests/unit/test_session.py -v

test-events:
	uv run pytest tests/unit/test_events.py -v

test-controller:
	uv run pytest tests/integration/test_controller.py -v

test-backend:
	uv run pytest tests/unit/test_backend_interface.py -v

test-config:
	uv run pytest tests/unit/test_config.py -v

test-benchmark:
	uv run pytest tests/unit/test_benchmark_registry.py tests/integration/test_benchmark_cli.py -v

# ============================================================================
# Code Quality
# ============================================================================

lint:
	uv run ruff check pentestgpt/ tests/

lint-fix:
	uv run ruff check --fix pentestgpt/ tests/

format:
	uv run ruff format pentestgpt/ tests/

format-check:
	uv run ruff format --check pentestgpt/ tests/

typecheck:
	uv run mypy pentestgpt/

check: lint typecheck
	@echo "All checks passed!"

# ============================================================================
# CI Simulation (End-to-End)
# ============================================================================

# Full CI simulation - mirrors GitHub Actions workflow exactly
ci:
	@echo "=========================================="
	@echo "Running full CI simulation..."
	@echo "=========================================="
	@echo ""
	@echo "[1/5] Lint check (ruff check)..."
	uv run ruff check pentestgpt/ tests/
	@echo ""
	@echo "[2/5] Format check (ruff format --check)..."
	uv run ruff format --check pentestgpt/ tests/
	@echo ""
	@echo "[3/5] Type check (mypy)..."
	uv run mypy pentestgpt/
	@echo ""
	@echo "[4/5] Running tests..."
	uv run pytest tests/ -v --ignore=tests/docker/
	@echo ""
	@echo "[5/5] Building package..."
	uv build
	@echo ""
	@echo "=========================================="
	@echo "CI simulation completed successfully!"
	@echo "=========================================="

# Quick CI - skip build step (faster iteration)
ci-quick:
	@echo "=========================================="
	@echo "Running quick CI simulation..."
	@echo "=========================================="
	@echo ""
	@echo "[1/4] Lint check (ruff check)..."
	uv run ruff check pentestgpt/ tests/
	@echo ""
	@echo "[2/4] Format check (ruff format --check)..."
	uv run ruff format --check pentestgpt/ tests/
	@echo ""
	@echo "[3/4] Type check (mypy)..."
	uv run mypy pentestgpt/
	@echo ""
	@echo "[4/4] Running tests..."
	uv run pytest tests/ -v --ignore=tests/docker/
	@echo ""
	@echo "=========================================="
	@echo "Quick CI simulation completed successfully!"
	@echo "=========================================="

# Full CI with Docker tests (requires Docker)
ci-full:
	@echo "=========================================="
	@echo "Running full CI simulation with Docker..."
	@echo "=========================================="
	@echo ""
	@echo "[1/7] Lint check (ruff check)..."
	uv run ruff check pentestgpt/ tests/
	@echo ""
	@echo "[2/7] Format check (ruff format --check)..."
	uv run ruff format --check pentestgpt/ tests/
	@echo ""
	@echo "[3/7] Type check (mypy)..."
	uv run mypy pentestgpt/
	@echo ""
	@echo "[4/7] Running tests..."
	uv run pytest tests/ -v --ignore=tests/docker/
	@echo ""
	@echo "[5/7] Validating docker-compose config..."
	docker compose config
	@echo ""
	@echo "[6/7] Building Docker image..."
	docker compose build
	@echo ""
	@echo "[7/7] Running Docker tests..."
	uv run pytest tests/docker/ -v -m docker
	@echo ""
	@echo "[8/8] Building package..."
	uv build
	@echo ""
	@echo "=========================================="
	@echo "Full CI with Docker completed successfully!"
	@echo "=========================================="

# ============================================================================
# Build
# ============================================================================

build:
	uv build

clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ============================================================================
# Local Development
# ============================================================================

# Run the TUI locally (for development)
run:
	uv run pentestgpt --target example.com

# Run in debug mode
run-debug:
	uv run pentestgpt --target example.com --debug

# Run in raw mode (no TUI, streaming output for debugging)
run-raw:
	uv run pentestgpt --target example.com --raw

# Watch for changes and run tests
watch:
	uv run ptw tests/ -- -v
