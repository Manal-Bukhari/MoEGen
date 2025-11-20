# MoEGen - Mixture-of-Experts Text Generator

An intelligent text generation system that uses **Mixture-of-Experts (MoE)** architecture to automatically route requests to specialized AI models based on content analysis.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg?logo=react)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**MoEGen** is a sophisticated text generation platform that leverages the **Mixture-of-Experts** machine learning architecture. Instead of using a single monolithic model, MoEGen intelligently routes user requests to specialized expert models, each trained and optimized for specific types of content.

### Key Concept: Mixture-of-Experts (MoE)

The MoE architecture consists of:
- **Multiple Expert Models**: Specialized models for different content types (Stories, Poems, Emails)
- **Intelligent Router**: Analyzes input and selects the most appropriate expert
- **Efficient Resource Usage**: Only activates relevant experts, saving computational resources
- **Better Quality**: Domain-specific experts produce higher quality output than generic models

## Features

### Intelligent Content Routing
- **Automatic Expert Selection**: Analyzes input context and selects the best expert
- **Multi-Expert Support**: Story, Poem, and Email specialists
- **Seamless Switching**: Dynamically routes to appropriate expert without user intervention

### Content Generation Capabilities
- **Story Expert**: Creative narratives, fiction, plot development
- **Poem Expert**: Poetry, verses, rhythmic content
- **Email Expert**: Professional emails, formal communications, business writing

### Modern Architecture
- **RESTful API**: Clean, well-documented backend endpoints
- **Responsive UI**: Modern React frontend with Vite
- **Containerized**: Docker and Docker Compose for easy deployment
- **Scalable**: Microservices-ready architecture

## Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.9+)
- **Architecture**: Mixture-of-Experts pattern
- **API Design**: RESTful with automatic OpenAPI documentation
- **Containerization**: Docker

### Frontend
- **Framework**: React 18+
- **Build Tool**: Vite
- **Styling**: Modern CSS
- **HTTP Client**: Axios
- **Web Server**: Nginx (production)

### DevOps
- **Orchestration**: Docker Compose
- **Reverse Proxy**: Nginx
- **Environment**: Multi-stage Docker builds

## Project Structure

```
MoEGen/
├── backend/
│   ├── experts/
│   │   ├── __init__.py
│   │   ├── base_expert.py          # Base expert class interface
│   │   ├── story_expert.py         # Story generation specialist
│   │   ├── poem_expert.py          # Poetry generation specialist
│   │   └── email_expert.py         # Email writing specialist
│   ├── routers/
│   │   ├── __init__.py
│   │   └── text_router.py          # Intelligent routing logic
│   ├── utils/                       # Utility functions
│   ├── main.py                      # FastAPI application entry
│   ├── requirements.txt             # Python dependencies
│   ├── Dockerfile                   # Backend container config
│   └── start.sh                     # Backend startup script
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ExpertSelector.jsx  # Expert selection UI
│   │   │   ├── TextGenerator.jsx   # Main generation interface
│   │   │   ├── OutputDisplay.jsx   # Generated text display
│   │   │   ├── Header.jsx          # App header
│   │   │   └── Footer.jsx          # App footer
│   │   ├── services/
│   │   │   └── api.js              # API service layer
│   │   ├── styles/
│   │   │   ├── App.css
│   │   │   └── index.css
│   │   ├── App.jsx                 # Main App component
│   │   └── main.jsx                # React entry point
│   ├── Dockerfile                   # Frontend container config
│   ├── nginx.conf                   # Nginx configuration
│   ├── package.json                 # Node dependencies
│   └── vite.config.js              # Vite build config
│
├── docker-compose.yml               # Multi-container orchestration
├── setup.sh                         # Setup automation script
├── .gitignore
├── ARCHITECTURE.md                  # Detailed architecture docs
├── PROJECT_SUMMARY.md               # Project summary
├── LICENSE
└── README.md
```

## Getting Started

### Prerequisites

**Option 1: Docker (Recommended)**
- Docker 20.10+
- Docker Compose 2.0+

**Option 2: Manual Setup**
- Python 3.9+
- Node.js 18+
- pip and npm

## Quick Start with Docker (Recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/Manal-Bukhari/MoEGen.git
cd MoEGen
```

### 2. Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

### 3. Start All Services

```bash
docker-compose up -d
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 5. Stop Services

```bash
docker-compose down
```

## Manual Setup (Without Docker)

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
# Navigate to frontend directory (in a new terminal)
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The application will be available at:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Backend Configuration
BACKEND_PORT=8000
BACKEND_HOST=0.0.0.0

# Frontend Configuration
FRONTEND_PORT=3000
VITE_API_BASE_URL=http://localhost:8000

# Model Configuration (Optional)
MODEL_TEMPERATURE=0.7
MAX_TOKENS=500
```

### Expert Model Configuration

Edit `backend/experts/base_expert.py` to customize expert behavior:

```python
class BaseExpert:
    def __init__(self, temperature=0.7, max_tokens=500):
        self.temperature = temperature
        self.max_tokens = max_tokens
```

## How It Works

### 1. User Input Processing

```
User Input → Text Router → Input Analysis
```

The router analyzes:
- Keywords and phrases
- Writing style indicators
- Content type signals
- Tone and formality

### 2. Expert Selection

```
Input Analysis → Expert Selection Logic → Selected Expert
```

The router uses heuristics or ML models to determine:
- Story Expert: Narrative elements, plot keywords, character mentions
- Poem Expert: Rhythmic patterns, emotional language, verse structures
- Email Expert: Professional tone, greeting/closing patterns, formal language

### 3. Content Generation

```
Selected Expert → AI Model → Generated Text → Post-processing → Output
```

Each expert:
- Uses specialized prompts
- Applies domain-specific templates
- Optimizes for content type
- Ensures quality and coherence

## Usage Examples

### Generate a Story

```javascript
// Frontend API call
const response = await generateText({
  prompt: "A brave knight embarks on a quest to find the lost kingdom",
  expertType: "story", // Optional: auto-detected if not specified
});
```

### Generate a Poem

```javascript
const response = await generateText({
  prompt: "Write about the beauty of autumn leaves",
  expertType: "poem",
});
```

### Generate an Email

```javascript
const response = await generateText({
  prompt: "Request a meeting with the project manager about deadline extension",
  expertType: "email",
});
```

## API Endpoints

### POST `/generate`

Generate text using the appropriate expert.

**Request:**
```json
{
  "prompt": "Your text prompt here",
  "expert_type": "story|poem|email",  // Optional
  "max_length": 500,                   // Optional
  "temperature": 0.7                   // Optional
}
```

**Response:**
```json
{
  "generated_text": "Generated content...",
  "expert_used": "story",
  "confidence": 0.95,
  "generation_time": 1.23
}
```

### GET `/experts`

List all available experts.

**Response:**
```json
{
  "experts": [
    {
      "name": "story",
      "description": "Specialized in creative narrative generation",
      "capabilities": ["fiction", "plot", "characters"]
    },
    {
      "name": "poem",
      "description": "Specialized in poetry and verse creation",
      "capabilities": ["rhyme", "rhythm", "imagery"]
    },
    {
      "name": "email",
      "description": "Specialized in professional email writing",
      "capabilities": ["formal", "business", "communication"]
    }
  ]
}
```

### GET `/health`

Check system health status.

## Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v
```

### Frontend Tests

```bash
cd frontend
npm run test
```

### Integration Tests

```bash
# Run with Docker Compose
docker-compose -f docker-compose.test.yml up
```

## Architecture Details

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

### Key Components

1. **Base Expert Class**: Abstract interface for all experts
2. **Specialized Experts**: Domain-specific implementations
3. **Text Router**: Intelligent routing algorithm
4. **API Layer**: FastAPI endpoints and request handling
5. **Frontend Components**: React UI for user interaction

### Design Patterns Used

- **Strategy Pattern**: Interchangeable expert strategies
- **Factory Pattern**: Expert instantiation
- **Singleton Pattern**: Router instance management
- **Repository Pattern**: API service abstraction

## Deployment

### Docker Compose (Production)

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Cloud Deployment

**AWS ECS / Azure Container Instances:**
```bash
# Build images
docker-compose build

# Push to registry
docker tag moegen-backend:latest your-registry/moegen-backend:latest
docker push your-registry/moegen-backend:latest
```

**Kubernetes:**
```bash
# Generate K8s manifests
kompose convert -f docker-compose.yml

# Deploy
kubectl apply -f moegen-deployment.yaml
```

## Security Considerations

- **API Rate Limiting**: Implemented to prevent abuse
- **Input Validation**: All inputs sanitized and validated
- **CORS Configuration**: Properly configured for production
- **Environment Secrets**: Use environment variables for sensitive data
- **Container Security**: Non-root users in Docker containers

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Commit your changes**
   ```bash
   git commit -m 'Add some amazing feature'
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 for Python code
- Use ESLint/Prettier for JavaScript
- Write meaningful commit messages
- Add documentation for new features
- Ensure all tests pass

## Roadmap

### Phase 1: Core Features (Completed)
- [x] Basic MoE architecture
- [x] Story, Poem, Email experts
- [x] Intelligent routing
- [x] React frontend
- [x] Docker deployment

### Phase 2: Enhancements (In Progress)
- [ ] Add more expert types (Code, Technical, Creative)
- [ ] Implement advanced routing with ML models
- [ ] User authentication and history
- [ ] Fine-tuning interface for experts
- [ ] Performance optimization

### Phase 3: Advanced Features (Planned)
- [ ] Multi-language support
- [ ] Custom expert training
- [ ] A/B testing framework
- [ ] Analytics dashboard
- [ ] Mobile application

## Known Issues

- Long generation times for complex prompts (>1000 words)
- Router accuracy can be improved with ML-based classification
- Limited expert model customization in UI

See [Issues](https://github.com/Manal-Bukhari/MoEGen/issues) for full list.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Manal Bukhari**
- Roll No: 22L-7917
- GitHub: [@Manal-Bukhari](https://github.com/Manal-Bukhari)
- Email: manal.bukhari@nu.edu.pk
- University: FAST-NUCES Lahore

## Acknowledgments

- FastAPI team for the excellent framework
- React community for frontend tools
- Docker for containerization platform
- OpenAI for inspiration on AI architectures
- FAST-NUCES for academic support

## References

- [Mixture-of-Experts Paper](https://arxiv.org/abs/1701.06538)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## Support

For questions or issues:
- **GitHub Issues**: [MoEGen Issues](https://github.com/Manal-Bukhari/MoEGen/issues)
- **Email**: manalbukhari2233@gmail.com

---

**Built with FastAPI, React, and Docker**

*Intelligent text generation through specialized expertise*
