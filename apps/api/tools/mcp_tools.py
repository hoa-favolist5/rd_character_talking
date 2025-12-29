"""MCP-compatible tools for CrewAI agents.

This module provides tool INTERFACES with MCP-compatible schemas.
All database operations are delegated to database.py for maintainability.

Architecture:
- database.py  = Database operations (SQL logic, single source of truth)
- mcp_tools.py = Tool interfaces (MCP schemas, delegates to database.py)
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from tools.database import (
    search_movies,
    load_conversation_history,
    format_conversation_history,
    _pool_cache,
    _pool_lock,
)

# Thread pool for running async code in sync context
_executor = ThreadPoolExecutor(max_workers=4)
_db_lock = threading.Lock()


def run_async(coro):
    """Run async coroutine in a separate thread."""
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
# MCP Tool 1: Movie Database Query
# =============================================================================

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
            # Delegate to database.py
            return run_async(search_movies(query, content_type, limit))
        except Exception as e:
            return f"Database search error: {str(e)}"


# =============================================================================
# MCP Tool 2: Conversation History
# =============================================================================

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
            # Delegate to database.py
            async def get_formatted_history():
                conversations = await load_conversation_history(session_id, limit)
                return format_conversation_history(conversations)
            
            return run_async(get_formatted_history())
        except Exception as e:
            return f"History retrieval error: {str(e)}"


# =============================================================================
# Factory function
# =============================================================================

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
