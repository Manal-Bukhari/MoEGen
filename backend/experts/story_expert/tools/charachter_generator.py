from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

llm_tool = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8)

class CharacterInput(BaseModel):
    archetype: str = Field(description="The role (e.g., 'Reluctant Hero').")
    setting: str = Field(description="The story world.")
    theme: str = Field(description="The central theme (e.g., 'Betrayal').")

@tool("character_generator", args_schema=CharacterInput)
def character_generator(archetype: str, setting: str, theme: str) -> str:
    """
    Generates a complex character profile with internal conflict and a growth arc.
    """
    prompt = f"""
    Create a deep, complex character profile.
    Archetype: {archetype}
    Setting: {setting}
    Theme: {theme}
    
    Output specific sections:
    1. Core Identity (Name, Age, Look)
    2. The Ghost (A past trauma or event haunting them)
    3. The Lie ( A misconception they have about themselves)
    4. The Want vs. The Need (External goal vs Internal growth)
    """
    response = llm_tool.invoke(prompt)
    return response.content