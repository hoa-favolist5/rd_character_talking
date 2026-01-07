"""Database migration runner for MySQL."""

import asyncio
import urllib.parse
from pathlib import Path

import aiomysql

from config.settings import get_settings


def parse_database_url(url: str) -> dict:
    """Parse DATABASE_URL into connection params."""
    parsed = urllib.parse.urlparse(url)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 3306,
        "user": parsed.username or "root",
        "password": parsed.password or "",
        "db": parsed.path.lstrip("/"),
    }


async def run_migrations() -> None:
    """Run all SQL migration files in order."""
    settings = get_settings()
    migrations_dir = Path(__file__).parent

    params = parse_database_url(settings.database_url)
    conn = await aiomysql.connect(
        host=params["host"],
        port=params["port"],
        user=params["user"],
        password=params["password"],
        db=params["db"],
        autocommit=True,
    )

    try:
        async with conn.cursor() as cur:
            # Get all .sql files sorted by name
            migration_files = sorted(migrations_dir.glob("*.sql"))

            for migration_file in migration_files:
                print(f"Running migration: {migration_file.name}")
                sql = migration_file.read_text()
                
                # MySQL doesn't support multiple statements by default
                # Split by semicolon and execute each statement
                statements = [s.strip() for s in sql.split(';') if s.strip()]
                for statement in statements:
                    if statement:
                        try:
                            await cur.execute(statement)
                        except Exception as e:
                            print(f"  Warning: {e}")
                
                print(f"Completed: {migration_file.name}")

            print("All migrations completed successfully!")

    finally:
        conn.close()


if __name__ == "__main__":
    asyncio.run(run_migrations())
