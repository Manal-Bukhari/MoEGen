from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from .. import config  # Import your config file

# Initialize tool-specific model
llm_tool = ChatGoogleGenerativeAI(
    api_key=config.GEMINI_API_KEY,
    model=config.GEMINI_MODEL,
    temperature=0.8,  # High creativity for characters
)

class CharacterInput(BaseModel):
    archetype: str = Field(description="The role (e.g., 'Reluctant Hero').")
    genre: str = Field(description="The story genre.")

@tool("character_generator", args_schema=CharacterInput)
def character_generator(archetype: str, genre: str) -> str:
    """Generates a deep character profile with internal motivations."""
    
    prompt = f"""
    Create a complex character for a {genre} story.
    Archetype: {archetype}
    
    Define:
    1. Name & Look
    2. The 'Lie' they believe about themselves
    3. The 'Ghost' (past trauma)
    4. The Goal
    """
    response = llm_tool.invoke(prompt)
    return response.content