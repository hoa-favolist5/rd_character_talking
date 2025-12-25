.PHONY: dev dev-api dev-web db-up db-down setup test lint clean

# Development
dev: db-up
	@echo "Starting development environment..."
	@make -j2 dev-api dev-web

dev-api:
	cd apps/api && source .venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000

dev-web:
	cd apps/web && npm run dev

# Database
db-up:
	docker-compose up -d postgres

db-down:
	docker-compose down

db-reset:
	docker-compose down -v
	docker-compose up -d postgres
	sleep 3
	cd apps/api && source .venv/bin/activate && python -m db.migrations.run

# Setup
setup: setup-api setup-web

setup-api:
	cd apps/api && python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

setup-web:
	cd apps/web && npm install

# Testing
test:
	cd apps/api && source .venv/bin/activate && pytest
	cd apps/web && npm run test

# Linting
lint:
	cd apps/api && source .venv/bin/activate && ruff check . && mypy .
	cd apps/web && npm run lint

# Clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name node_modules -exec rm -rf {} +
	find . -type d -name .nuxt -exec rm -rf {} +

# Production
build:
	cd apps/web && npm run build
	cd apps/api && docker build -t character-api .

deploy:
	./infrastructure/scripts/deploy.sh

