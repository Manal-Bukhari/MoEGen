from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from .. import config

llm_tool = ChatGoogleGenerativeAI(
    api_key=config.GEMINI_API_KEY,
    model=config.GEMINI_MODEL,
    temperature=0.5,  # Lower temp for structured planning
)

class PlotInput(BaseModel):
    character_profile: str = Field(description="The main character details.")
    genre: str = Field(description="The story genre.")
    prompt_template: str = Field(description="The prompt template to use.")

@tool("plot_outliner", args_schema=PlotInput)
def plot_outliner(character_profile: str, genre: str, prompt_template: str) -> str:
    """Generates a 5-point plot outline."""
    
    # We inject the dynamic values into the prompt passed from the agent
    formatted_prompt = prompt_template.format(
        genre=genre,
        original_query="Generated from Character Profile",
        character_profile=character_profile
    )
    
    response = llm_tool.invoke(formatted_prompt)
    return response.content