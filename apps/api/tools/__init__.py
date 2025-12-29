# CrewAI tools module

# MCP-compatible tools for agents
from tools.mcp_tools import (
    MCPMovieDatabaseQueryTool,
    MCPRestaurantDatabaseQueryTool,
    MCPConversationHistoryTool,
    get_mcp_tools,
)

# Database operations (single source of truth for all SQL logic)
from tools.database import (
    # Connection management
    get_db_pool,
    run_async,
    # Query functions
    search_movies,
    search_restaurants,
    load_conversation_history,
    format_conversation_history,
    # Tools
    SaveConversationTool,
)

__all__ = [
    # MCP Tools (interfaces)
    "MCPMovieDatabaseQueryTool",
    "MCPRestaurantDatabaseQueryTool",
    "MCPConversationHistoryTool",
    "get_mcp_tools",
    # Database operations
    "get_db_pool",
    "run_async",
    "search_movies",
    "search_restaurants",
    "load_conversation_history",
    "format_conversation_history",
    "SaveConversationTool",
]
