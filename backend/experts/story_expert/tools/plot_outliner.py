from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

llm_tool = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

class PlotInput(BaseModel):
    character_profile: str = Field(description="The detailed character profile.")
    genre: str = Field(description="The genre.")

@tool("plot_outliner", args_schema=PlotInput)
def plot_outliner(character_profile: str, genre: str) -> str:
    """
    Generates a 5-beat plot structure (Hook, Inciting Incident, Rising Action, Climax, Resolution).
    """
    prompt = f"""
    Create a tight plot outline for a {genre} story based on this character:
    {character_profile}
    
    Format strictly as:
    [HOOK]: ...
    [INCITING INCIDENT]: ...
    [RISING ACTION]: ...
    [CLIMAX]: ...
    [RESOLUTION]: ...
    """
    response = llm_tool.invoke(prompt)
    return response.content