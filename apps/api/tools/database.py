"""Database tools for CrewAI agents."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import asyncpg
from asyncpg import Pool
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from config.settings import get_settings

# Thread pool for running async code in sync context
_executor = ThreadPoolExecutor(max_workers=4)  # Increase workers for better parallelism
_db_lock = threading.Lock()  # Lock for database operations

# Thread-local storage for connection pools (one pool per event loop)
_pool_cache: dict[int, Pool] = {}
_pool_lock = threading.Lock()


async def get_db_pool() -> Pool:
    """Get or create a connection pool for the current event loop.
    
    Uses a cached pool per event loop to avoid creating new connections
    for each operation while respecting asyncpg's event loop requirements.
    """
    loop = asyncio.get_event_loop()
    loop_id = id(loop)
    
    with _pool_lock:
        if loop_id in _pool_cache:
            pool = _pool_cache[loop_id]
            # Check if pool is still valid
            if not pool._closed:
                return pool
    
    # Create new pool for this event loop
    settings = get_settings()
    pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=2,
        max_size=5,
        command_timeout=30,
    )
    
    with _pool_lock:
        _pool_cache[loop_id] = pool
    
    return pool


async def get_db_connection():
    """Get a connection from the pool.
    
    This is more efficient than creating a new connection each time.
    """
    pool = await get_db_pool()
    return await pool.acquire()


async def release_db_connection(pool: Pool, conn):
    """Release a connection back to the pool."""
    await pool.release(conn)


def run_async(coro):
    """Run async coroutine in a new event loop in a separate thread with locking."""
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            # Clean up pool for this loop when done
            loop_id = id(loop)
            with _pool_lock:
                if loop_id in _pool_cache:
                    pool = _pool_cache.pop(loop_id)
                    if not pool._closed:
                        loop.run_until_complete(pool.close())
            loop.close()
    
    # Use lock to ensure only one DB operation at a time
    with _db_lock:
        future = _executor.submit(run_in_thread)
        return future.result(timeout=30)


class DatabaseQueryInput(BaseModel):
    """Input schema for database query tool."""

    query: str = Field(description="Keyword or question to search for")
    content_type: str | None = Field(default=None, description="Filter by content type (movie/tv, etc.)")
    limit: int = Field(default=10, description="Maximum number of results to retrieve")


class DatabaseQueryTool(BaseTool):
    """Tool for querying the movie database using keyword search."""

    name: str = "movie_database_query"
    description: str = """
    Searches the movie/TV show database for relevant information.
    Use this when you need information to answer user questions.
    Performs keyword search across titles, overviews, taglines, etc.
    """
    args_schema: type[BaseModel] = DatabaseQueryInput

    def _run(self, query: str, content_type: str | None = None, limit: int = 10) -> str:
        """Execute the database query."""
        try:
            return run_async(self._async_run(query, content_type, limit))
        except Exception as e:
            return f"Database search error: {str(e)}"

    async def _async_run(self, query: str, content_type: str | None = None, limit: int = 10) -> str:
        """Async implementation of database query using keyword search."""
        pool = None
        conn = None
        try:
            # Get connection from pool
            pool = await get_db_pool()
            conn = await pool.acquire()

            # First, check if table exists and has data
            try:
                check_sql = """
                    SELECT COUNT(*) as total FROM data_archive_movie_master
                """
                count_result = await conn.fetchrow(check_sql)
                total_count = count_result['total'] if count_result else 0
                print(f"[DB DEBUG] Table has {total_count} total records")

                if total_count == 0:
                    return "Database table 'data_archive_movie_master' is empty. Please import movie data first."
            except Exception as table_err:
                print(f"[DB ERROR] Table check failed: {table_err}")
                return f"Database error: Table 'data_archive_movie_master' may not exist. Error: {table_err}"

            # Search in title AND overview for better results
            sql = """
                SELECT 
                    id,
                    title,
                    original_title,
                    overview,
                    content_type,
                    release_date,
                    vote_average,
                    vote_count,
                    genre_ids,
                    tagline,
                    runtime,
                    status
                FROM data_archive_movie_master
                WHERE 
                    title ILIKE '%' || $1 || '%' OR overview ILIKE '%' || $1 || '%'
                ORDER BY vote_average DESC NULLS LAST, vote_count DESC NULLS LAST
                LIMIT $2
            """
            print(f"[DB DEBUG] Searching for: '{query}' with limit {limit}")
            results = await conn.fetch(sql, query, limit)
            print(f"[DB DEBUG] Found {len(results)} results")

            if not results:
                # Try to show sample titles to help debug
                sample_sql = "SELECT title FROM data_archive_movie_master LIMIT 5"
                samples = await conn.fetch(sample_sql)
                sample_titles = [r['title'] for r in samples]
                return f"No results found for '{query}'. Sample titles in DB: {sample_titles}"
            
            formatted = ["[Search Results]\n"]
            for i, r in enumerate(results, 1):
                runtime_str = f"{r['runtime']} min" if r['runtime'] else "Unknown"
                vote_str = f"{r['vote_average']}" if r['vote_average'] else "N/A"
                formatted.append(
                    f"{i}. Title: {r['title']}\n"
                    f"   Original Title: {r['original_title'] or '-'}\n"
                    f"   Type: {r['content_type'] or '-'}\n"
                    f"   Release Date: {r['release_date'] or '-'}\n"
                    f"   Runtime: {runtime_str}\n"
                    f"   Rating: {vote_str} ({r['vote_count'] or 0} votes)\n"
                    f"   Overview: {r['overview'][:200] + '...' if r['overview'] and len(r['overview']) > 200 else r['overview'] or '-'}\n"
                )

            return "\n".join(formatted)

        except Exception as e:
            import traceback
            error_detail = f"{type(e).__name__}: {str(e)}"
            print(f"[DB ERROR] {error_detail}")
            print(traceback.format_exc())
            return f"Database search error: {error_detail}"
        finally:
            # Release connection back to pool instead of closing
            if conn and pool:
                await pool.release(conn)


class ConversationHistoryInput(BaseModel):
    """Input schema for conversation history tool."""

    session_id: str = Field(description="Session ID")
    limit: int = Field(default=10, description="Maximum number of conversations to retrieve")


class ConversationHistoryTool(BaseTool):
    """Tool for retrieving conversation history."""

    name: str = "conversation_history"
    description: str = """
    Retrieves past conversation history.
    Use this to review previous interactions with the user
    and generate contextually appropriate responses.
    """
    args_schema: type[BaseModel] = ConversationHistoryInput

    def _run(self, session_id: str, limit: int = 10) -> str:
        """Execute the history retrieval."""
        try:
            return run_async(self._async_run(session_id, limit))
        except Exception as e:
            return f"History retrieval error: {str(e)}"

    async def _async_run(self, session_id: str, limit: int = 10) -> str:
        """Async implementation of history retrieval."""
        pool = None
        conn = None
        try:
            # Get connection from pool
            pool = await get_db_pool()
            conn = await pool.acquire()

            sql = """
                SELECT user_message, ai_response, created_at
                FROM conversations
                WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """

            results = await conn.fetch(sql, session_id, limit)

            if not results:
                return "No past conversation history. This is the first conversation."

            formatted = ["[Conversation History]\n"]
            for r in reversed(results):
                formatted.append(
                    f"User: {r['user_message']}\n"
                    f"Assistant: {r['ai_response']}\n"
                    f"---"
                )

            return "\n".join(formatted)

        except Exception as e:
            return f"History retrieval error: {str(e)}"
        finally:
            # Release connection back to pool
            if conn and pool:
                await pool.release(conn)


class SaveConversationInput(BaseModel):
    """Input schema for saving conversation."""

    session_id: str = Field(description="Session ID")
    user_message: str = Field(description="User's message")
    ai_response: str = Field(description="AI's response")
    user_emotion: str | None = Field(default=None, description="User's emotion")
    response_emotion: str | None = Field(default=None, description="Response emotion")
    audio_url: str | None = Field(default=None, description="Audio file URL")


class SaveConversationTool(BaseTool):
    """Tool for saving conversation to database."""

    name: str = "save_conversation"
    description: str = """
    Saves the conversation to the database.
    Use this after the conversation is complete.
    """
    args_schema: type[BaseModel] = SaveConversationInput

    def _run(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        user_emotion: str | None = None,
        response_emotion: str | None = None,
        audio_url: str | None = None,
    ) -> str:
        """Execute the save operation."""
        try:
            return run_async(
                self._async_run(
                    session_id, user_message, ai_response,
                    user_emotion, response_emotion, audio_url
                )
            )
        except Exception as e:
            return f"Save error: {str(e)}"

    async def _async_run(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        user_emotion: str | None = None,
        response_emotion: str | None = None,
        audio_url: str | None = None,
    ) -> str:
        """Async implementation of save operation."""
        pool = None
        conn = None
        try:
            # Get connection from pool
            pool = await get_db_pool()
            conn = await pool.acquire()

            sql = """
                INSERT INTO conversations 
                (session_id, user_message, ai_response, user_emotion, response_emotion, audio_url)
                VALUES ($1, $2, $3, $4, $5, $6)
            """

            await conn.execute(
                sql,
                session_id,
                user_message,
                ai_response,
                user_emotion,
                response_emotion,
                audio_url,
            )

            return "Conversation saved."

        except Exception as e:
            return f"Save error: {str(e)}"
        finally:
            # Release connection back to pool
            if conn and pool:
                await pool.release(conn)
