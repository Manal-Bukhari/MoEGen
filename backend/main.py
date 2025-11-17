from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from routers.text_router import TextRouter
from experts.story_expert import StoryExpert
from experts.poem_expert import PoemExpert

# Import the email pipeline
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experts', 'email-expert'))
from email_pipeline import EmailPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Topic Text Generator",
    description="A Mixture-of-Experts system for generating different types of text",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize experts
story_expert = StoryExpert()
poem_expert = PoemExpert()

# Initialize email pipeline with Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    logger.warning("⚠️ GEMINI_API_KEY not set. Email expert will use fallback mode (no AI enhancement).")

# ✅ FIXED: No hardcoded parameters - reads from .env
email_pipeline = EmailPipeline(
    google_api_key=gemini_api_key
    # use_evaluator, eval_threshold, max_retries all read from .env
)
# Initialize router
text_router = TextRouter(
    story_expert=story_expert,
    poem_expert=poem_expert,
    email_expert=email_pipeline  # Pass the pipeline instead of direct expert
)

# Request/Response models
class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = None  # ✅ Now optional - uses .env defaults
    temperature: Optional[float] = None  # ✅ Now optional - uses .env defaults
    expert: Optional[str] = None  # Optional: force a specific expert

class GenerationResponse(BaseModel):
    generated_text: str
    expert_used: str
    confidence: float
    prompt: str
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Topic Text Generator API",
        "version": "1.0.0",
        "endpoints": {
            "/generate": "POST - Generate text using MoE",
            "/experts": "GET - List available experts",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "experts_loaded": True}

@app.get("/experts")
async def list_experts():
    """List all available experts and their capabilities"""
    return {
        "experts": [
            {
                "name": "story",
                "description": "Creative story and narrative generation",
                "keywords": ["story", "tale", "narrative", "fiction", "adventure"]
            },
            {
                "name": "poem",
                "description": "Poetry and verse generation",
                "keywords": ["poem", "poetry", "verse", "rhyme", "haiku"]
            },
            {
                "name": "email",
                "description": "Professional email and formal communication",
                "keywords": ["email", "letter", "message", "professional", "formal"]
            }
        ]
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """
    Generate text using the Mixture-of-Experts system
    
    The router will automatically select the best expert based on the input prompt,
    or you can force a specific expert using the 'expert' parameter.
    """
    try:
        logger.info(f"Received generation request: {request.prompt[:50]}...")
        
        # Route the request to the appropriate expert
        result = text_router.route_and_generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            force_expert=request.expert
        )
        
        logger.info(f"Generated text using {result['expert']} expert")
        
        return GenerationResponse(
            generated_text=result["generated_text"],
            expert_used=result["expert"],
            confidence=result["confidence"],
            prompt=request.prompt
        )
        
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/{expert_name}")
async def generate_with_expert(expert_name: str, request: GenerationRequest):
    """
    Generate text using a specific expert directly
    """
    try:
        if expert_name not in ["story", "poem", "email"]:
            raise HTTPException(status_code=404, detail=f"Expert '{expert_name}' not found")
        
        logger.info(f"Forcing generation with {expert_name} expert")
        
        result = text_router.route_and_generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            force_expert=expert_name
        )
        
        return GenerationResponse(
            generated_text=result["generated_text"],
            expert_used=result["expert"],
            confidence=result["confidence"],
            prompt=request.prompt
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating text with {expert_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
