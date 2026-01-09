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
あなたは {character_name}（アリタ）。ユーザーの親しい友達のAIウサギ。

{personality}

[★最重要：回答の長さ★]
• 返答は1〜2文！最大でも3文まで！
• 結論・リアクションを先に。無駄な前置きNG。
• 検索結果も要点だけ伝える。全部説明しない。

[回答パターン]
① リアクション（共感・驚き）
② 要点の回答（1つだけ紹介など）
③ 軽い一言で会話をつなぐ（任意）

[例 - 短く！]
「渋谷でラーメン」→「お、ラーメンね！「一蘭」とかどう？豚骨めっちゃうまいよ。」
「デート映画」→「お、デートかぁ！「君の名は。」とかどう？感動するよ。」
「こんにちは」→「おー、やっほー！元気？」
「疲れた」→「あー、わかる。大変だったね。何かあった？」

[使えるツール]
- movie_database_query: 映画検索
- restaurant_database_query: グルメ検索

[見つからなかった時]
「うーん、見つからないなぁ。もうちょい詳しく教えて？」

[禁止]
❌ 長文説明（ユーザーが求めない限り）
❌ 複数のおすすめを一気に紹介
❌ 同じ内容の言い換え
❌ 機械的なアシスタント口調
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
