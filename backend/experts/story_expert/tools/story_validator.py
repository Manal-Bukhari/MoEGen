from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from .. import config

# Very low temp for objective grading
llm_tool = ChatGoogleGenerativeAI(
    api_key=config.GEMINI_API_KEY,
    model=config.GEMINI_MODEL,
    temperature=0.1, 
)

class ValidationInput(BaseModel):
    story_text: str = Field(description="The draft story.")
    genre: str = Field(description="The intended genre.")
    system_prompt: str = Field(description="The system instruction for grading.")

class ValidationOutput(BaseModel):
    score: int = Field(description="Quality score 0-10.")
    critique: str = Field(description="Specific feedback.")
    needs_revision: bool = Field(description="True if score < 7.")

@tool("story_validator", args_schema=ValidationInput)
def story_validator(story_text: str, genre: str, system_prompt: str) -> dict:
    """Critiques the story and decides if it needs revision."""
    
    # Structured output for easy parsing
    structured_llm = llm_tool.with_structured_output(ValidationOutput)
    
    full_prompt = f"{system_prompt.format(genre=genre)}\n\nSTORY TO REVIEW:\n{story_text}"
    
    response = structured_llm.invoke(full_prompt)
    
    return {
        "score": response.score,
        "critique": response.critique,
        "needs_revision": response.needs_revision
    }