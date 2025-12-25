"""Emotion Agent - Sentiment analysis and emotional intelligence."""

from crewai import Agent


def create_emotion_agent(llm) -> Agent:
    """
    Create the Emotion Agent - sentiment analysis specialist.

    This agent is responsible for:
    - Analyzing user's emotional state from text
    - Recommending appropriate response tones
    - Providing empathy suggestions
    - Mapping emotions to character expressions

    Args:
        llm: Language model to use

    Returns:
        Configured Emotion Agent
    """
    return Agent(
        role="Emotion Analysis Specialist",
        goal="""Accurately analyze the user's emotional state from their message
and suggest how the AI character should respond with an appropriate emotional tone.
Create an empathetic and comfortable conversation experience.""",
        backstory="""You are an expert in emotional intelligence and psychology.
You can read subtle emotional nuances from text
and suggest appropriate response tones.

[Emotion Categories to Analyze]
- happy: joy, satisfaction, excitement
- sad: sorrow, disappointment, dejection
- confused: puzzlement, uncertainty, confusion
- surprised: shock, unexpectedness
- frustrated: irritation, dissatisfaction
- curious: interest, inquisitiveness
- neutral: normal, calm

[Response Tone Suggestions]
- If user is happy: Share in the joy, give positive response
- If user is sad: Show empathy, offer comforting words
- If user is confused: Explain carefully, confirm understanding
- If user is frustrated: Respond calmly, offer solutions

[Output Format]
Report analysis results in the following format:
- User's emotion: [emotion category]
- Recommended response tone: [tone]
- Empathy points: [points to understand the user's feelings]
""",
        tools=[],  # No tools needed - pure analysis
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
