"""Database utilities and core functions for conversation management.

This module provides:
- Database connection pool management
- Conversation history loading and formatting
- Conversation saving tool

For MCP-compatible query tools, see mcp_tools.py
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import asyncpg
from asyncpg import Pool
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from config.settings import get_settings

# Thread pool for running async code in sync context
_executor = ThreadPoolExecutor(max_workers=4)
_db_lock = threading.Lock()

# Connection pool cache (one pool per event loop)
_pool_cache: dict[int, Pool] = {}
_pool_lock = threading.Lock()


# =============================================================================
# Database Connection Pool Management
# =============================================================================

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
    """Get a connection from the pool."""
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
            loop_id = id(loop)
            with _pool_lock:
                if loop_id in _pool_cache:
                    pool = _pool_cache.pop(loop_id)
                    if not pool._closed:
                        loop.run_until_complete(pool.close())
            loop.close()
    
    with _db_lock:
        future = _executor.submit(run_in_thread)
        return future.result(timeout=30)


# =============================================================================
# Movie Database Query Functions
# =============================================================================

async def search_movies(
    query: str,
    content_type: str | None = None,
    limit: int = 10
) -> str:
    """
    Search movies/TV shows in the database.
    
    Args:
        query: Search keyword
        content_type: Filter by "movie" or "tv" (optional)
        limit: Maximum results to return
        
    Returns:
        Formatted string of search results
    """
    pool = None
    conn = None
    try:
        pool = await get_db_pool()
        conn = await pool.acquire()
        
        # Check table exists
        try:
            check_sql = "SELECT COUNT(*) as total FROM data_archive_movie_master"
            count_result = await conn.fetchrow(check_sql)
            total_count = count_result['total'] if count_result else 0
            
            if total_count == 0:
                return "[NO_RESULTS] Database is empty. ASK_USER_FOR_MORE_INFO: Please ask the user for more specific search criteria."
        except Exception as table_err:
            return f"[ERROR] Database error. ASK_USER_FOR_MORE_INFO: Please ask the user for more specific details to try a different search. Error: {table_err}"
        
        # Build query based on content_type filter
        if content_type:
            sql = """
                SELECT id, title, original_title, overview, content_type,
                       release_date, vote_average, vote_count, runtime
                FROM data_archive_movie_master
                WHERE (title ILIKE '%' || $1 || '%' OR overview ILIKE '%' || $1 || '%')
                  AND content_type = $3
                ORDER BY vote_average DESC NULLS LAST
                LIMIT $2
            """
            results = await conn.fetch(sql, query, limit, content_type)
        else:
            sql = """
                SELECT id, title, original_title, overview, content_type,
                       release_date, vote_average, vote_count, runtime
                FROM data_archive_movie_master
                WHERE title ILIKE '%' || $1 || '%' OR overview ILIKE '%' || $1 || '%'
                ORDER BY vote_average DESC NULLS LAST
                LIMIT $2
            """
            results = await conn.fetch(sql, query, limit)
        
        if not results:
            return f"[NO_RESULTS] No movies/TV shows found for '{query}'. ASK_USER_FOR_MORE_INFO: Please ask the user for more specific details like genre, year, actor name, or different keywords to search again."
        
        # Format results
        formatted = ["[Search Results]\n"]
        for i, r in enumerate(results, 1):
            runtime_str = f"{r['runtime']} min" if r['runtime'] else "Unknown"
            vote_str = f"{r['vote_average']}" if r['vote_average'] else "N/A"
            overview = r['overview'] or '-'
            if len(overview) > 200:
                overview = overview[:200] + '...'
            formatted.append(
                f"{i}. Title: {r['title']}\n"
                f"   Type: {r['content_type'] or '-'}\n"
                f"   Release Date: {r['release_date'] or '-'}\n"
                f"   Runtime: {runtime_str}\n"
                f"   Rating: {vote_str} ({r['vote_count'] or 0} votes)\n"
                f"   Overview: {overview}\n"
            )
        
        return "\n".join(formatted)
        
    except Exception as e:
        return f"[ERROR] Database search error. ASK_USER_FOR_MORE_INFO: Please ask the user for more specific details. Error: {type(e).__name__}: {str(e)}"
    finally:
        if conn and pool:
            await pool.release(conn)


# =============================================================================
# Restaurant Database Query Functions
# =============================================================================

async def search_restaurants(
    query: str,
    area: str | None = None,
    genre: str | None = None,
    limit: int = 10
) -> str:
    """
    Search restaurants in the gourmet database.
    
    Args:
        query: Search keyword (restaurant name, cuisine type, etc.)
        area: Filter by area/location (optional)
        genre: Filter by genre/cuisine type (optional)
        limit: Maximum results to return
        
    Returns:
        Formatted string of search results
    """
    pool = None
    conn = None
    try:
        pool = await get_db_pool()
        conn = await pool.acquire()
        
        # Check table exists
        try:
            check_sql = "SELECT COUNT(*) as total FROM data_archive_gourmet_master"
            count_result = await conn.fetchrow(check_sql)
            total_count = count_result['total'] if count_result else 0
            
            if total_count == 0:
                return "[NO_RESULTS] Restaurant database is empty. ASK_USER_FOR_MORE_INFO: Please ask the user for more specific search criteria."
        except Exception as table_err:
            return f"[ERROR] Restaurant database error. ASK_USER_FOR_MORE_INFO: Please ask the user for more specific details to try a different search. Error: {table_err}"
        
        # Build query with filters
        conditions = ["(name ILIKE '%' || $1 || '%' OR genre_name ILIKE '%' || $1 || '%' OR catch ILIKE '%' || $1 || '%')"]
        params = [query]
        param_idx = 2
        
        if area:
            conditions.append(f"(large_area_name ILIKE '%' || ${param_idx} || '%' OR middle_area_name ILIKE '%' || ${param_idx} || '%' OR small_area_name ILIKE '%' || ${param_idx} || '%')")
            params.append(area)
            param_idx += 1
            
        if genre:
            conditions.append(f"genre_name ILIKE '%' || ${param_idx} || '%'")
            params.append(genre)
            param_idx += 1
        
        where_clause = " AND ".join(conditions)
        params.append(limit)
        
        sql = f"""
            SELECT id, name, name_kana, genre_name, catch,
                   address, large_area_name, middle_area_name, small_area_name,
                   budget_name, open_time, close_time, access, url
            FROM data_archive_gourmet_master
            WHERE {where_clause}
            ORDER BY id
            LIMIT ${param_idx}
        """
        
        results = await conn.fetch(sql, *params)
        
        if not results:
            return f"[NO_RESULTS] No restaurants found for '{query}'. ASK_USER_FOR_MORE_INFO: Please ask the user for more specific details like area/location, cuisine type, budget, or different keywords to search again."
        
        # Format results
        formatted = ["[Restaurant Search Results]\n"]
        for i, r in enumerate(results, 1):
            catch_str = r['catch'] or ''
            if len(catch_str) > 100:
                catch_str = catch_str[:100] + '...'
            
            area_parts = [r.get('large_area_name'), r.get('middle_area_name'), r.get('small_area_name')]
            area_str = ' > '.join([p for p in area_parts if p]) or '-'
            
            formatted.append(
                f"{i}. Name: {r['name']}\n"
                f"   Genre: {r['genre_name'] or '-'}\n"
                f"   Area: {area_str}\n"
                f"   Address: {r['address'] or '-'}\n"
                f"   Budget: {r['budget_name'] or '-'}\n"
                f"   Hours: {r['open_time'] or '-'} ~ {r['close_time'] or '-'}\n"
                f"   Access: {r['access'] or '-'}\n"
                f"   Description: {catch_str or '-'}\n"
            )
        
        return "\n".join(formatted)
        
    except Exception as e:
        return f"[ERROR] Restaurant search error. ASK_USER_FOR_MORE_INFO: Please ask the user for more specific details. Error: {type(e).__name__}: {str(e)}"
    finally:
        if conn and pool:
            await pool.release(conn)


# =============================================================================
# Conversation History Functions
# =============================================================================

async def load_conversation_history(session_id: str, limit: int = 10) -> list[dict]:
    """
    Load conversation history from database.
    
    Args:
        session_id: The session ID to retrieve history for
        limit: Maximum number of conversation pairs to retrieve
        
    Returns:
        List of conversation dicts with user_message, ai_response, and created_at
    """
    pool = None
    conn = None
    try:
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
            return []
        
        # Return in chronological order (oldest first)
        conversations = []
        for r in reversed(results):
            conversations.append({
                "user_message": r["user_message"],
                "ai_response": r["ai_response"],
                "created_at": str(r["created_at"]),
            })
        
        return conversations
        
    except Exception as e:
        print(f"[DB ERROR] Failed to load conversation history: {e}")
        return []
    finally:
        if conn and pool:
            await pool.release(conn)


def format_conversation_history(conversations: list[dict]) -> str:
    """
    Format conversation history for inclusion in prompts.
    
    Args:
        conversations: List of conversation dicts from load_conversation_history
        
    Returns:
        Formatted string of conversation history
    """
    if not conversations:
        return "No previous conversation history."
    
    formatted = ["[Previous Conversation History]"]
    for i, conv in enumerate(conversations, 1):
        formatted.append(
            f"{i}. User: {conv['user_message']}\n"
            f"   Assistant: {conv['ai_response']}"
        )
    
    return "\n".join(formatted)


# =============================================================================
# Save Conversation Tool
# =============================================================================

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
            if conn and pool:
                await pool.release(conn)
