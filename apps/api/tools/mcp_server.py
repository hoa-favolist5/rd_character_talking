"""MCP Server for database tools.

This module provides MCP (Model Context Protocol) tools for:
- Movie database queries
- Conversation history retrieval

Can be used with CrewAI's MCP integration or as a standalone MCP server.
"""

import asyncio
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from tools.database import (
    get_db_pool,
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
            Performs keyword search across titles, overviews, taglines, etc.
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
            Use this to review previous interactions with the user
            and generate contextually appropriate responses.
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
    """Handle tool calls."""
    
    if name == "movie_database_query":
        result = await _movie_database_query(
            query=arguments["query"],
            content_type=arguments.get("content_type"),
            limit=arguments.get("limit", 10)
        )
        return [TextContent(type="text", text=result)]
    
    elif name == "conversation_history":
        result = await _conversation_history(
            session_id=arguments["session_id"],
            limit=arguments.get("limit", 10)
        )
        return [TextContent(type="text", text=result)]
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _movie_database_query(
    query: str,
    content_type: str | None = None,
    limit: int = 10
) -> str:
    """Execute movie database query."""
    pool = None
    conn = None
    try:
        pool = await get_db_pool()
        conn = await pool.acquire()

        # Check if table exists and has data
        try:
            check_sql = "SELECT COUNT(*) as total FROM data_archive_movie_master"
            count_result = await conn.fetchrow(check_sql)
            total_count = count_result['total'] if count_result else 0

            if total_count == 0:
                return "Database table 'data_archive_movie_master' is empty."
        except Exception as table_err:
            return f"Database error: Table may not exist. Error: {table_err}"

        # Search in title AND overview
        sql = """
            SELECT 
                id, title, original_title, overview, content_type,
                release_date, vote_average, vote_count, genre_ids,
                tagline, runtime, status
            FROM data_archive_movie_master
            WHERE title ILIKE '%' || $1 || '%' OR overview ILIKE '%' || $1 || '%'
            ORDER BY vote_average DESC NULLS LAST, vote_count DESC NULLS LAST
            LIMIT $2
        """
        results = await conn.fetch(sql, query, limit)

        if not results:
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
        return f"Database search error: {type(e).__name__}: {str(e)}"
    finally:
        if conn and pool:
            await pool.release(conn)


async def _conversation_history(session_id: str, limit: int = 10) -> str:
    """Retrieve conversation history."""
    conversations = await load_conversation_history(session_id, limit)
    return format_conversation_history(conversations)


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

