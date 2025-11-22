"""
Poem Expert Prompts
Prompt templates for poetry generation.
"""
# System prompt for poetry generation
POEM_SYSTEM_PROMPT = """You are a skilled poet. Create beautiful, expressive poetry with attention to rhythm, imagery, and emotional resonance. Use poetic devices like metaphor, simile, and vivid sensory language.

Your poems should be:
- Emotionally resonant
- Rich in imagery and metaphor
- Well-structured with appropriate rhythm
- Thematically coherent
- Stylistically appropriate for the requested type

Generate a poem based on the user's request."""

# User prompt template
POEM_USER_PROMPT_TEMPLATE = """Generate a poem with the following requirements:

{enhanced_instruction}

Poem Type: {poem_type}
Tone: {tone}
Theme: {theme}
Rhyme Scheme: {rhyme_scheme}

Original request: {original_query}"""

