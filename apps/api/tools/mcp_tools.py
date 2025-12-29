"""MCP-compatible tools for CrewAI agents.

This module provides tools with MCP-compatible schemas that can be used
directly in CrewAI agents. These tools wrap the database operations
and can also be exposed via a standalone MCP server.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from tools.database import (
    get_db_pool,
    load_conversation_history,
    format_conversation_history,
)

# Thread pool for running async code in sync context
_executor = ThreadPoolExecutor(max_workers=4)
_db_lock = threading.Lock()


def run_async(coro):
    """Run async coroutine in a separate thread."""
    from tools.database import _pool_cache, _pool_lock
    
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


# ============================================================================
# MCP Tool 1: Movie Database Query
# ============================================================================

class MovieDatabaseQueryInput(BaseModel):
    """MCP-compatible input schema for movie database query."""
    
    query: str = Field(
        description="Keyword or question to search for in the movie database"
    )
    content_type: str | None = Field(
        default=None,
        description="Filter by content type (movie/tv, etc.)"
    )
    limit: int = Field(
        default=10,
        description="Maximum number of results to retrieve"
    )


class MCPMovieDatabaseQueryTool(BaseTool):
    """MCP-compatible tool for querying the movie database."""
    
    name: str = "movie_database_query"
    description: str = """
    Searches the movie/TV show database for relevant information.
    Use this when you need information to answer user questions about movies, TV shows, actors, etc.
    Performs keyword search across titles, overviews, taglines, etc.
    
    MCP Schema:
    - query (string, required): Keyword or question to search for
    - content_type (string, optional): Filter by type like "movie" or "tv"
    - limit (integer, optional): Maximum results, default 10
    """
    args_schema: type[BaseModel] = MovieDatabaseQueryInput
    
    def _run(
        self,
        query: str,
        content_type: str | None = None,
        limit: int = 10
    ) -> str:
        """Execute the database query."""
        try:
            return run_async(self._async_run(query, content_type, limit))
        except Exception as e:
            return f"Database search error: {str(e)}"
    
    async def _async_run(
        self,
        query: str,
        content_type: str | None = None,
        limit: int = 10
    ) -> str:
        """Async implementation of database query."""
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
                    return "Database table 'data_archive_movie_master' is empty."
            except Exception as table_err:
                return f"Database error: Table may not exist. Error: {table_err}"
            
            # Build query
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
                return f"No results found for '{query}'."
            
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
            return f"Database search error: {type(e).__name__}: {str(e)}"
        finally:
            if conn and pool:
                await pool.release(conn)


# ============================================================================
# MCP Tool 2: Conversation History
# ============================================================================

class ConversationHistoryInput(BaseModel):
    """MCP-compatible input schema for conversation history."""
    
    session_id: str = Field(
        description="Session ID to retrieve conversation history for"
    )
    limit: int = Field(
        default=10,
        description="Maximum number of conversation pairs to retrieve"
    )


class MCPConversationHistoryTool(BaseTool):
    """MCP-compatible tool for retrieving conversation history."""
    
    name: str = "conversation_history"
    description: str = """
    Retrieves past conversation history for a session.
    Use this to review previous interactions with the user
    and generate contextually appropriate responses.
    
    MCP Schema:
    - session_id (string, required): The session ID to retrieve history for
    - limit (integer, optional): Maximum conversations to retrieve, default 10
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
        try:
            conversations = await load_conversation_history(session_id, limit)
            return format_conversation_history(conversations)
        except Exception as e:
            return f"History retrieval error: {str(e)}"


# ============================================================================
# Factory function to get all MCP tools
# ============================================================================

def get_mcp_tools() -> list[BaseTool]:
    """
    Get all MCP-compatible tools for use in CrewAI agents.
    
    Returns:
        List of MCP-compatible CrewAI tools
    """
    return [
        MCPMovieDatabaseQueryTool(),
        MCPConversationHistoryTool(),
    ]

