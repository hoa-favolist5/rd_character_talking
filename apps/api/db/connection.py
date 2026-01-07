import urllib.parse
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import aiomysql

from config.settings import get_settings


class Database:
    """Async MySQL database connection manager."""

    def __init__(self) -> None:
        self._pool: aiomysql.Pool | None = None
        self._settings = get_settings()

    def _parse_database_url(self) -> dict:
        """Parse DATABASE_URL into connection params."""
        url = self._settings.database_url
        # mysql://user:pass@host:port/dbname
        parsed = urllib.parse.urlparse(url)
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 3306,
            "user": parsed.username or "root",
            "password": parsed.password or "",
            "db": parsed.path.lstrip("/"),
        }

    async def connect(self) -> None:
        """Create database connection pool."""
        if self._pool is not None:
            return

        params = self._parse_database_url()
        self._pool = await aiomysql.create_pool(
            host=params["host"],
            port=params["port"],
            user=params["user"],
            password=params["password"],
            db=params["db"],
            minsize=2,
            maxsize=10,
            autocommit=True,
        )

    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[aiomysql.Cursor, None]:
        """Acquire a connection and cursor from the pool."""
        if self._pool is None:
            raise RuntimeError("Database not connected")

        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                yield cur

    async def execute(self, query: str, *args) -> int:
        """Execute a query."""
        async with self.acquire() as cur:
            await cur.execute(query, args if args else None)
            return cur.rowcount

    async def fetch(self, query: str, *args) -> list[dict]:
        """Fetch multiple rows."""
        async with self.acquire() as cur:
            await cur.execute(query, args if args else None)
            return await cur.fetchall()

    async def fetchrow(self, query: str, *args) -> dict | None:
        """Fetch a single row."""
        async with self.acquire() as cur:
            await cur.execute(query, args if args else None)
            return await cur.fetchone()

    async def fetchval(self, query: str, *args):
        """Fetch a single value."""
        async with self.acquire() as cur:
            await cur.execute(query, args if args else None)
            row = await cur.fetchone()
            return list(row.values())[0] if row else None


# Global database instance
db = Database()


async def get_db() -> Database:
    """Get database instance (dependency injection)."""
    return db
