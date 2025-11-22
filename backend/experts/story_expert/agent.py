from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# --- IMPORTS ---
from . import config
from .prompts import (
    STORY_SYSTEM_PROMPT, 
    STORY_USER_PROMPT_TEMPLATE, 
    PLANNER_PROMPT, 
    VALIDATOR_SYSTEM_PROMPT
)
from .tools import character_generator, plot_outliner, story_validator

# --- STATE DEFINITION ---
class StoryState(TypedDict):
    original_query: str
    enhanced_instruction: Optional[str]
    genre: str
    tone: str
    
    # Internal Workflow Data
    character_profile: Optional[str]
    plot_outline: Optional[str]
    current_draft: Optional[str]
    critique: Optional[str]
    revision_count: int
    
    # Final Output
    final_story: str

# --- MODEL INIT (Using Config) ---
llm = ChatGoogleGenerativeAI(
    api_key=config.GEMINI_API_KEY,
    model=config.GEMINI_MODEL,
    temperature=config.TEMPERATURE,
    max_output_tokens=config.MAX_TOKENS,
)

# --- NODES ---

def node_planner(state: StoryState):
    """Step 1: Generate Character & Outline"""
    print("--- PLANNER: Architecting Story ---")
    genre = state.get("genre") or config.DEFAULT_GENRE
    
    # 1. Generate Character
    char_profile = character_generator.invoke({
        "archetype": "Protagonist", 
        "genre": genre
    })
    
    # 2. Generate Outline (Using the PLANNER_PROMPT from prompts.py)
    outline = plot_outliner.invoke({
        "character_profile": char_profile,
        "genre": genre,
        "prompt_template": PLANNER_PROMPT
    })
    
    return {
        "character_profile": char_profile, 
        "plot_outline": outline, 
        "revision_count": 0
    }

def node_writer(state: StoryState):
    """Step 2: Write the Draft"""
    print(f"--- WRITER: Drafting (Revision {state['revision_count']}) ---")
    
    # We construct the 'key_elements' field using our Plan + Any Critique
    plan_context = f"STORY PLAN:\n{state['plot_outline']}\n\nCHARACTER:\n{state['character_profile']}"
    
    if state.get("critique"):
        plan_context += f"\n\nCRITICAL FEEDBACK TO FIX:\n{state['critique']}"

    # Use YOUR specific User Prompt Template
    prompt = STORY_USER_PROMPT_TEMPLATE.format(
        enhanced_instruction=state.get("enhanced_instruction") or state["original_query"],
        genre=state.get("genre") or config.DEFAULT_GENRE,
        tone=state.get("tone") or config.DEFAULT_TONE,
        key_elements=plan_context, # Injecting the plan here
        length_preference="Standard",
        original_query=state["original_query"]
    )
    
    messages = [SystemMessage(content=STORY_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    return {"current_draft": response.content, "revision_count": state["revision_count"] + 1}

def node_reviewer(state: StoryState):
    """Step 3: Quality Control"""
    print("--- REVIEWER: Validating ---")
    
    result = story_validator.invoke({
        "story_text": state["current_draft"],
        "genre": state.get("genre") or config.DEFAULT_GENRE,
        "system_prompt": VALIDATOR_SYSTEM_PROMPT
    })
    
    return {
        "critique": result["critique"], 
        "needs_revision": result["needs_revision"],
        "final_story": state["current_draft"] # Update final story tentatively
    }

# --- EDGES ---

def should_revise(state: StoryState):
    # Stop if good enough OR if we have tried 2 times already
    if state["needs_revision"] and state["revision_count"] < 2:
        return "revise"
    return "publish"

# --- GRAPH ---
workflow = StateGraph(StoryState)

workflow.add_node("planner", node_planner)
workflow.add_node("writer", node_writer)
workflow.add_node("reviewer", node_reviewer)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "writer")
workflow.add_edge("writer", "reviewer")

workflow.add_conditional_edges(
    "reviewer",
    should_revise,
    {
        "revise": "writer",
        "publish": END
    }
)

story_expert = workflow.compile()