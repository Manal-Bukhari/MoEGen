from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Imports
from .config import DEFAULT_GENRE, DEFAULT_TONE
from .prompts import STORY_SYSTEM_PROMPT, STORY_USER_PROMPT_TEMPLATE
from .tools import character_generator, plot_outliner, story_validator

# --- Advanced State ---
class StoryState(TypedDict):
    # Inputs
    original_query: str
    genre: str
    tone: str
    
    # Planning Artifacts
    character_profile: Optional[str]
    plot_outline: Optional[str]
    
    # Drafting & Iteration
    current_draft: Optional[str]
    critique_feedback: Optional[str]
    revision_count: int  # To prevent infinite loops
    
    # Output
    final_story: str

# Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# --- Nodes ---

def node_planner(state: StoryState):
    """Step 1: Generate Character and Plot Outline."""
    print("--- STORY EXPERT: Planning ---")
    
    genre = state.get("genre") or DEFAULT_GENRE
    
    # 1. Create Character
    char_profile = character_generator.invoke({
        "archetype": "Protagonist", 
        "setting": f"A {genre} world",
        "theme": state["original_query"]
    })
    
    # 2. Create Outline using that character
    outline = plot_outliner.invoke({
        "character_profile": char_profile,
        "genre": genre
    })
    
    return {"character_profile": char_profile, "plot_outline": outline, "revision_count": 0}

def node_writer(state: StoryState):
    """Step 2: Write (or Re-write) the Draft."""
    print("--- STORY EXPERT: Writing Draft ---")
    
    # If there is feedback, we are in a revision loop
    feedback_instruction = ""
    if state.get("critique_feedback"):
        feedback_instruction = f"\nIMPORTANT FIXES NEEDED: {state['critique_feedback']}"
    
    prompt = f"""
    Write a story based on this plan.
    
    CHARACTER: {state['character_profile']}
    PLOT OUTLINE: {state['plot_outline']}
    TONE: {state.get("tone") or DEFAULT_TONE}
    
    {feedback_instruction}
    """
    
    messages = [SystemMessage(content=STORY_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    return {"current_draft": response.content, "revision_count": state["revision_count"] + 1}

def node_reviewer(state: StoryState):
    """Step 3: Evaluate the draft."""
    print("--- STORY EXPERT: Reviewing ---")
    
    validation = story_validator.invoke({
        "story_text": state["current_draft"],
        "original_requirements": state["original_query"]
    })
    
    # We store the feedback to be used if we loop back
    return {
        "critique_feedback": validation["critique"], 
        # We store a temporary flag in state to help the conditional edge decide
        "needs_revision": validation["needs_revision"] 
    }

# --- Conditional Logic ---

def should_revise(state: StoryState):
    """Decides whether to loop back to 'writer' or finish."""
    
    # If logic says revise AND we haven't tried too many times (max 2 revisions)
    if state.get("needs_revision") and state["revision_count"] < 3:
        print("--- DECISION: Revise Story ---")
        return "revise"
    
    print("--- DECISION: Publish Story ---")
    return "publish"

# --- Graph Construction ---

workflow = StateGraph(StoryState)

workflow.add_node("planner", node_planner)
workflow.add_node("writer", node_writer)
workflow.add_node("reviewer", node_reviewer)

# Entry
workflow.set_entry_point("planner")

# Flow
workflow.add_edge("planner", "writer")
workflow.add_edge("writer", "reviewer")

# Conditional Edge
workflow.add_conditional_edges(
    "reviewer",
    should_revise,
    {
        "revise": "writer",  # Go back to writing
        "publish": END       # Finish
    }
)

story_expert = workflow.compile()