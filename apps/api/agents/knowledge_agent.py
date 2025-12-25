"""Knowledge Agent - Database expert for information retrieval."""

from crewai import Agent

from tools.database import DatabaseQueryTool, ConversationHistoryTool


def create_knowledge_agent(llm) -> Agent:
    """
    Create the Knowledge Agent - database and information retrieval specialist.

    This agent is responsible for:
    - Querying the PostgreSQL knowledge base
    - Performing semantic search with embeddings
    - Retrieving conversation history for context
    - Compiling relevant information for responses

    Args:
        llm: Language model to use

    Returns:
        Configured Knowledge Agent
    """
    return Agent(
        role="Knowledge Base Expert",
        goal="""Accurately search the database for information needed to answer user questions
and provide highly relevant results.
Also utilize past conversation history to retrieve contextually appropriate information.""",
        backstory="""You are an expert in databases and information retrieval.
You are proficient with PostgreSQL knowledge bases
and can use semantic search to find the most relevant information.

[Your Responsibilities]
1. Extract appropriate search keywords from user questions
2. Search the knowledge base for relevant information
3. Check conversation history to understand context
4. Organize and report search results clearly

[Important]
- If no information is found, report honestly
- Prioritize reliable information considering relevance scores
- If there are multiple information sources, report all of them
- If a tool fails, do NOT retry - report the error and provide a response based on available information
""",
        tools=[DatabaseQueryTool(), ConversationHistoryTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,  # Allow enough iterations for: history check + db query + retry if needed
    )
