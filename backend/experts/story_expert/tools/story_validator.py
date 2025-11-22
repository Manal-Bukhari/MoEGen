from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal

llm_tool = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

class ValidationInput(BaseModel):
    story_text: str = Field(description="The draft story.")
    original_requirements: str = Field(description="What the user asked for.")

class ValidationOutput(BaseModel):
    score: int = Field(description="Score out of 10.")
    reasoning: str = Field(description="Critique details.")
    needs_revision: bool = Field(description="True if score < 7.")

@tool("story_validator", args_schema=ValidationInput)
def story_validator(story_text: str, original_requirements: str) -> dict:
    """
    Reviews the story and returns a structured score and critique.
    """
    # We use structured output (JSON mode) to ensure the agent can parse the result
    structured_llm = llm_tool.with_structured_output(ValidationOutput)
    
    prompt = f"""
    Rate this story based on these requirements: {original_requirements}
    
    Story:
    {story_text}
    
    If the plot is weak or requirements are missed, score it low.
    If it is excellent, score high.
    """
    response = structured_llm.invoke(prompt)
    
    # Return as dict for the state graph
    return {
        "score": response.score,
        "critique": response.reasoning,
        "needs_revision": response.needs_revision
    }