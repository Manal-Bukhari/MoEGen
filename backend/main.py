from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import LangGraph agents
# from experts.story_expert.agent import StoryExpertAgent
from experts.poem_expert.agent import PoemExpertAgent
from experts.email_expert.agent import EmailExpertAgent
from routers.text_router import TextRouter

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

# Initialize LangGraph agents
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    logger.warning("⚠️ GEMINI_API_KEY not set.")

# story_expert_agent = StoryExpertAgent()  # TODO: Implement when ready
poem_expert_agent = PoemExpertAgent()  # TODO: Implement when ready
email_expert_agent = EmailExpertAgent()

# Initialize router
text_router = TextRouter(
    story_expert=None,  # TODO: Replace with story_expert_agent when ready
    poem_expert=poem_expert_agent,  # TODO: Replace with poem_expert_agent when ready
    email_expert=email_expert_agent
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
    # TODO: Implement when LangGraph agents are ready
    if text_router is None:
        raise HTTPException(
            status_code=503, 
            detail="Service not ready: LangGraph agents are not yet implemented"
        )
    
    try:
        logger.info(f"📥 Received generation request")
        logger.info(f"   Prompt: {request.prompt[:100]}...")
        logger.debug(f"   max_length: {request.max_length}, temperature: {request.temperature}, expert: {request.expert}")
        
        # Route the request to the appropriate expert
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
        logger.info(f"   Generated text length: {len(result['generated_text'])} chars")
        logger.debug(f"   Generated text preview: {result['generated_text'][:150]}...")
        
        # Build response
        response = GenerationResponse(
            generated_text=result["generated_text"],
            expert_used=result["expert"],
            confidence=result["confidence"],
            prompt=request.prompt
        )
        logger.info(f"📤 Returning response to client")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error generating text: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/{expert_name}")
async def generate_with_expert(expert_name: str, request: GenerationRequest):
    """
    Generate text using a specific expert directly
    """
    # Check if router is ready
    if text_router is None:
        raise HTTPException(
            status_code=503, 
            detail="Service not ready: Router not initialized"
        )
    
    try:
        if expert_name not in ["story", "poem", "email"]:
            raise HTTPException(status_code=404, detail=f"Expert '{expert_name}' not found")
        
        # Only email and poem expert is implemented
        if expert_name not in ["email", "poem"]:
            raise HTTPException(
            status_code=503,
            detail=f"Expert '{expert_name}' not yet implemented. Available: email, poem"
        )
        
        logger.info(f"📥 Received forced generation request for {expert_name} expert")
        logger.info(f"   Prompt: {request.prompt[:100]}...")
        logger.debug(f"   max_length: {request.max_length}, temperature: {request.temperature}")
        
        logger.info(f"🔄 Forcing generation with {expert_name} expert...")
        result = text_router.route_and_generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            force_expert=expert_name
        )
        
        logger.info(f"✅ Generation complete: {result['expert']} expert, {len(result['generated_text'])} chars")
        logger.info(f"📤 Returning response to client")
        
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
