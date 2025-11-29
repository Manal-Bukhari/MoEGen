"""
Main FastAPI application for Multi-Topic Text Generator
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import LangGraph agents
from experts.story_expert.agent import StoryExpertAgent
from experts.poem_expert.agent import PoemExpertAgent
from experts.email_expert.agent import EmailExpertAgent
from routers.text_router import TextRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Topic Text Generator",
    description="A Mixture-of-Experts system for generating stories, poems, and emails using LangGraph",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verify API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    logger.error("❌ GEMINI_API_KEY not set in environment variables!")
    logger.error("   Please add GEMINI_API_KEY to your .env file")
else:
    logger.info(f"✅ GEMINI_API_KEY loaded (starts with: {gemini_api_key[:20]}...)")

# Initialize LangGraph agents
logger.info("🚀 Initializing expert agents...")

try:
    logger.info("   Initializing Story Expert...")
    story_expert_agent = StoryExpertAgent()  # ✅ INITIALIZED
    logger.info("   ✅ Story Expert initialized")
except Exception as e:
    logger.error(f"   ❌ Story Expert initialization failed: {e}")
    story_expert_agent = None


story_expert_agent = StoryExpertAgent()  # TODO: Implement when ready
poem_expert_agent = PoemExpertAgent()  # TODO: Implement when ready
email_expert_agent = EmailExpertAgent()
try:
    logger.info("   Initializing Email Expert...")
    email_expert_agent = EmailExpertAgent()
    logger.info("   ✅ Email Expert initialized")
except Exception as e:
    logger.error(f"   ❌ Email Expert initialization failed: {e}")
    email_expert_agent = None

# Initialize router with all available experts
logger.info("🔄 Initializing TextRouter...")
text_router = TextRouter(

    story_expert=story_expert_agent,   # ✅ PASSED TO ROUTER
    poem_expert=poem_expert_agent,  # TODO: Replace with poem_expert_agent when ready
    email_expert=email_expert_agent
)
logger.info("✅ TextRouter initialized")

# Request/Response models
class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="The input prompt for text generation")
    max_length: Optional[int] = Field(None, description="Maximum length (uses .env defaults if not specified)")
    temperature: Optional[float] = Field(None, description="Temperature for generation (uses .env defaults if not specified)")
    expert: Optional[str] = Field(None, description="Force a specific expert: 'story', 'poem', or 'email'")

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Write a short sci-fi story about an AI discovering emotions",
                "max_length": 4000,
                "temperature": 0.8,
                "expert": "story"
            }
        }


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str = Field(..., description="The generated text")
    expert_used: str = Field(..., description="Which expert was used")
    confidence: float = Field(..., description="Routing confidence score")
    routing_method: str = Field(..., description="How the expert was selected")
    prompt: str = Field(..., description="The original prompt")
    all_scores: Dict[str, int] = Field(..., description="Keyword scores for all experts")


class ExpertInfo(BaseModel):
    """Information about an expert."""
    name: str
    description: str
    keywords: list[str]
    available: bool


# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    active_experts = text_router.get_expert_info()["active_experts"]

    return {
        "message": "Multi-Topic Text Generator API",
        "version": "2.0.0",
        "active_experts": active_experts,
        "endpoints": {
            "/generate": "POST - Generate text using automatic expert routing",
            "/generate/{expert_name}": "POST - Generate text using a specific expert",
            "/experts": "GET - List all available experts",
            "/router/info": "GET - Get router configuration info",
            "/health": "GET - Health check"
        },
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    router_info = text_router.get_expert_info()

    return {
        "status": "healthy",
        "experts_loaded": len(router_info["active_experts"]),
        "available_experts": router_info["available_experts"],
        "active_experts": router_info["active_experts"]
    }


@app.get("/experts", response_model=list[ExpertInfo])
async def list_experts():
    """List all available experts and their capabilities."""
    router_info = text_router.get_expert_info()
    active_experts = router_info["active_experts"]

    experts_data = [
        {
            "name": "story",
            "description": "Creative story and narrative generation with character development and plot structure",
            "keywords": ["story", "tale", "narrative", "fiction", "adventure", "fantasy", "sci-fi"],
            "available": "story" in active_experts
        },
        {
            "name": "poem",
            "description": "Poetry and verse generation with various styles (haiku, sonnet, free verse)",
            "keywords": ["poem", "poetry", "verse", "rhyme", "haiku", "sonnet"],
            "available": "poem" in active_experts
        },
        {
            "name": "email",
            "description": "Professional email and formal communication with proper structure",
            "keywords": ["email", "letter", "message", "professional", "formal", "leave request"],
            "available": "email" in active_experts
        }
    ]

    return experts_data


@app.get("/router/info")
async def get_router_info():
    """Get detailed information about the router configuration."""
    return text_router.get_expert_info()


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """
    Generate text using automatic expert routing.

    The router will analyze your prompt and select the most appropriate expert
    (story, poem, or email) based on the content.
    """
    try:
        logger.info(f"📥 Received generation request")
        logger.info(f"   Prompt: {request.prompt[:100]}...")
        logger.debug(f"   Parameters: max_length={request.max_length}, temperature={request.temperature}, expert={request.expert}")

        # Route and generate
        logger.info("🔄 Routing request to expert...")
        result = text_router.route_and_generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            force_expert=request.expert
        )

        logger.info(f"✅ Generation complete")
        logger.info(f"   Expert used: {result['expert']}")
        logger.info(f"   Confidence: {result['confidence']:.1%}")
        logger.info(f"   Method: {result['routing_method']}")
        logger.info(f"   Generated text length: {len(result['generated_text'])} chars")

        # Build response
        response = GenerationResponse(
            generated_text=result["generated_text"],
            expert_used=result["expert"],
            confidence=result["confidence"],
            routing_method=result["routing_method"],
            prompt=request.prompt,
            all_scores=result["all_scores"]
        )

        logger.info(f"📤 Returning response to client")
        return response

    except ValueError as e:
        logger.error(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error generating text: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate/{expert_name}", response_model=GenerationResponse)
async def generate_with_expert(expert_name: str, request: GenerationRequest):
    """
    Generate text using a specific expert directly.

    Available experts:
    - story: For creative narratives and fiction
    - poem: For poetry and verse
    - email: For professional communication
    """
    try:
        # Validate expert name
        valid_experts = ["story", "poem", "email"]
        if expert_name not in valid_experts:
            raise HTTPException(
                status_code=404,
                detail=f"Expert '{expert_name}' not found. Available: {valid_experts}"
            )

        # Check if expert is active
        active_experts = text_router.get_expert_info()["active_experts"]
        if expert_name not in active_experts:
            raise HTTPException(
                status_code=503,
                detail=f"Expert '{expert_name}' is not yet implemented or failed to load. "
                       f"Available experts: {active_experts}"
            )

        logger.info(f"📥 Received forced generation request for {expert_name} expert")
        logger.info(f"   Prompt: {request.prompt[:100]}...")

        # Generate with forced expert
        logger.info(f"🔄 Forcing generation with {expert_name} expert...")
        result = text_router.route_and_generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            force_expert=expert_name
        )

        logger.info(f"✅ Generation complete: {len(result['generated_text'])} chars")
        logger.info(f"📤 Returning response to client")

        return GenerationResponse(
            generated_text=result["generated_text"],
            expert_used=result["expert"],
            confidence=result["confidence"],
            routing_method=result["routing_method"],
            prompt=request.prompt,
            all_scores=result["all_scores"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating text with {expert_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("="*60)
    logger.info("🚀 Multi-Topic Text Generator API Starting")
    logger.info("="*60)

    router_info = text_router.get_expert_info()
    logger.info(f"✅ Active Experts: {router_info['active_experts']}")
    logger.info(f"📊 Keyword Counts: {router_info['keyword_counts']}")

    logger.info("="*60)
    logger.info("🌐 API Ready - Visit http://localhost:8000/docs for interactive documentation")
    logger.info("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information."""
    logger.info("👋 Multi-Topic Text Generator API Shutting Down")


if __name__ == "__main__":
    import uvicorn

    # Run the application
    logger.info("Starting Uvicorn server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )