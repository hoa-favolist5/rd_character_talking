import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from asyncpg import Pool

from config.settings import get_settings


class Database:
    """Async PostgreSQL database connection manager."""

    def __init__(self) -> None:
        self._pool: Pool | None = None
        self._settings = get_settings()

    async def connect(self) -> None:
        """Create database connection pool."""
        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(
            self._settings.database_url,
            min_size=2,
            max_size=10,
            command_timeout=60,
        )

    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Acquire a connection from the pool."""
        if self._pool is None:
            raise RuntimeError("Database not connected")

        async with self._pool.acquire() as connection:
            yield connection

    async def execute(self, query: str, *args) -> str:
        """Execute a query."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list[asyncpg.Record]:
        """Fetch multiple rows."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> asyncpg.Record | None:
        """Fetch a single row."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        """Fetch a single value."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)


# Global database instance
db = Database()


async def get_db() -> Database:
    """Get database instance (dependency injection)."""
    return db

