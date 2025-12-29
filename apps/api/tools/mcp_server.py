"""Standalone MCP Server for database tools.

This module provides a standalone MCP (Model Context Protocol) server
that can be run externally for integrations with other MCP clients.

All database operations are delegated to database.py.

Usage:
    python -m tools.mcp_server
"""

import asyncio
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from tools.database import (
    search_movies,
    load_conversation_history,
    format_conversation_history,
)

# Create MCP server instance
server = Server("database-mcp-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available database tools."""
    return [
        Tool(
            name="movie_database_query",
            description="""
            Searches the movie/TV show database for relevant information.
            Use this when you need information to answer user questions about movies, TV shows, actors, etc.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword or question to search for"
                    },
                    "content_type": {
                        "type": "string",
                        "description": "Filter by content type (movie/tv, etc.)",
                        "default": None
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to retrieve",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="conversation_history",
            description="""
            Retrieves past conversation history for a session.
            Use this to review previous interactions with the user.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to retrieve history for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of conversations to retrieve",
                        "default": 10
                    }
                },
                "required": ["session_id"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls - delegates to database.py functions."""
    
    if name == "movie_database_query":
        # Delegate to database.py
        result = await search_movies(
            query=arguments["query"],
            content_type=arguments.get("content_type"),
            limit=arguments.get("limit", 10)
        )
        return [TextContent(type="text", text=result)]
    
    elif name == "conversation_history":
        # Delegate to database.py
        conversations = await load_conversation_history(
            session_id=arguments["session_id"],
            limit=arguments.get("limit", 10)
        )
        result = format_conversation_history(conversations)
        return [TextContent(type="text", text=result)]
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# Entry point for running as standalone MCP server
async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
