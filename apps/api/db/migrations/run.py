"""Database migration runner."""

import asyncio
from pathlib import Path

import asyncpg

from config.settings import get_settings


async def run_migrations() -> None:
    """Run all SQL migration files in order."""
    settings = get_settings()
    migrations_dir = Path(__file__).parent

    conn = await asyncpg.connect(settings.database_url)

    try:
        # Get all .sql files sorted by name
        migration_files = sorted(migrations_dir.glob("*.sql"))

        for migration_file in migration_files:
            print(f"Running migration: {migration_file.name}")
            sql = migration_file.read_text()
            await conn.execute(sql)
            print(f"Completed: {migration_file.name}")

        print("All migrations completed successfully!")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_migrations())

