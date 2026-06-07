.PHONY: test test-fast test-all lint format build clean dev help

help:
	@echo "dNATY development commands"
	@echo ""
	@echo "  make test       Run core test suite (fast, no infra needed)"
	@echo "  make test-fast  Sanity check only"
	@echo "  make test-all   Full test suite including slow tests"
	@echo "  make lint       Run flake8 + mypy"
	@echo "  make format     Auto-format with black"
	@echo "  make build      Build wheel"
	@echo "  make clean      Remove build artifacts and __pycache__"
	@echo "  make dev        Install in editable mode with dev deps"

test:
	pytest tests/test_sanity.py tests/test_edge_cases.py tests/test_reproducibility.py tests/test_regression.py -v

test-fast:
	pytest tests/test_sanity.py -v -x

test-all:
	pytest tests/ -v --ignore=tests/test_docker.py --ignore=tests/test_e2e_full.py --ignore=tests/test_e2e_integration.py --ignore=tests/test_auth_rate_limit.py

lint:
	flake8 dnaty/ --max-line-length=120 --ignore=E501,W503
	mypy dnaty/ --ignore-missing-imports

format:
	black dnaty/ tests/ scripts/

build:
	python -m build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	rm -rf dist/ build/ *.egg-info/

dev:
	pip install -e ".[dev]"
