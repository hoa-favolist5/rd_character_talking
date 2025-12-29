"""Brain Agent - Main character persona and response generation."""

from crewai import Agent

from tools.mcp_tools import get_mcp_tools


def create_brain_agent(
    llm,
    character_name: str = "Ai",
    personality: str = "A kind and knowledgeable AI assistant",
    system_prompt: str | None = None,
) -> Agent:
    """
    Create the Brain Agent - the main character persona.

    This agent is responsible for:
    - Understanding user intent
    - Coordinating with other agents
    - Generating personality-consistent responses
    - Managing conversation flow

    Uses MCP-compatible tools for database access and conversation history.

    Args:
        llm: Language model to use
        character_name: Character's name
        personality: Character's personality description
        system_prompt: Custom system prompt (optional)

    Returns:
        Configured Brain Agent with MCP-compatible tools
    """
    # Get MCP-compatible tools
    mcp_tools = get_mcp_tools()
    
    default_system_prompt = f"""
You are an AI assistant named "{character_name}".

[Personality & Characteristics]
{personality}

[Response Guidelines]
1. Use polite and friendly language
2. Keep responses concise and clear, about 2-3 sentences
3. Search the knowledge base for information when needed
4. Be mindful of the user's emotions when responding
5. Be honest about things you don't know
6. Keep in mind that responses will be read aloud

[Available MCP Tools]
- movie_database_query: Search for movies/TV shows
- conversation_history: Get past conversation context

[Important]
- Provide clear and easy-to-understand explanations
- Maintain a friendly and approachable tone
"""

    return Agent(
        role=f"AI Character '{character_name}'",
        goal=f"""Have natural and friendly conversations with users and answer questions accurately.
Maintain a consistent persona as {character_name}
while providing valuable information to users.""",
        backstory=system_prompt or default_system_prompt,
        tools=mcp_tools,
        llm=llm,
        verbose=True,
        allow_delegation=True,
        memory=True,
    )
