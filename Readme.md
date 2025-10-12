# Synergistic PyTorch RAG Assistant (SPRA)

**SPRA is a context-aware AI developer tool, delivered as a Chrome Extension, designed to provide expert-level assistance for the PyTorch deep learning framework. It uses a novel hybrid RAG architecture to deliver accurate, conceptually rich, and citation-backed answers.**

This project moves beyond traditional RAG by fusing multiple advanced techniques to create a more robust and intelligent system that can act as a learning tool, a debugging assistant, and a library navigator.

## 🚀 Features

- **🧠 Advanced RAG Pipeline**: Multi-query generation, RAG-Fusion, step-back prompting, and HyDE
- **🔄 Hybrid Retrieval**: Combines vector search (ChromaDB) with graph search (Neo4j)
- **🎯 Intelligent Routing**: Context-aware query classification and routing
- **📚 Knowledge Graph**: Captures PyTorch module relationships and dependencies
- **💬 Context-Aware Chat**: Multi-turn conversations with page context integration
- **📊 Comprehensive Evaluation**: RAGAS metrics for continuous improvement
- **🔧 Chrome Extension**: Seamless integration with PyTorch documentation

## 🏗️ Architecture

### Synergistic Hybrid RAG Pipeline

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Query Input   │───▶│  Query Router    │───▶│ Query Transformer│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Simple Vector    │    │ Multi-Query     │
                       │ Search           │    │ Generation      │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────────────────────────────┐
                       │         Hybrid Retrieval               │
                       │  ┌─────────────┐  ┌─────────────────┐  │
                       │  │  Vector     │  │  Graph Search   │  │
                       │  │  Search     │  │  (Neo4j)        │  │
                       │  │ (ChromaDB)  │  │                 │  │
                       │  └─────────────┘  └─────────────────┘  │
                       └─────────────────────────────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Re-ranking     │
                       │ (Cross-Encoder)  │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Generation     │
                       │  (Self-RAG)      │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Final Answer     │
                       │ + Citations      │
                       └──────────────────┘
```

## 🛠️ Tech Stack

### Backend
- **Python 3.9+** with **FastAPI** for REST API
- **LangChain** for RAG pipeline orchestration
- **ChromaDB** for vector storage and similarity search
- **Neo4j** for knowledge graph storage
- **Ollama** for local LLM inference
- **Sentence-Transformers** for embeddings
- **RAGAS** for evaluation metrics

### Frontend
- **Chrome Extension** (Manifest V3)
- **JavaScript** with modern ES6+ features
- **CSS3** with responsive design
- **WebSocket** support for real-time chat

### Infrastructure
- **Docker** for containerization
- **Redis** for caching (optional)
- **Structured logging** with Python logging

## 📦 Installation & Setup

### Prerequisites

1. **Python 3.9+**
2. **Node.js 16+** (for development)
3. **Neo4j** (Docker or local installation)
4. **Ollama** (for local LLM)
5. **Chrome Browser** (for extension)

### 1. Backend Setup

```bash
# Clone the repository
git clone <repository-url>
cd Synergy-A-Pytorch-Doc-Assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
    cd backend
    pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize databases
python -m backend.stores.vector_store
python -m backend.stores.graph_store

# Start the API server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Database Setup

#### Neo4j (Docker)
```bash
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_PLUGINS=["apoc"] \
    neo4j:latest
```

#### ChromaDB
ChromaDB will be automatically initialized when you run the application.

### 3. Local LLM Setup

#### Install Ollama
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

#### Pull Models
```bash
# Pull recommended models
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull codellama:7b
```

### 4. Chrome Extension Setup

```bash
# Navigate to Chrome extensions
# chrome://extensions/

# Enable Developer mode
# Click "Load unpacked"
# Select the chrome_extension directory
```

### 5. Data Population

```bash
# Scrape PyTorch documentation
python -m backend.scrapers.pytorch_scraper

# Process and index documents
python -m backend.processors.doc_processor

# Populate knowledge graph
python -m backend.stores.graph_store --populate
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
USE_LOCAL_MODEL=true

# Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
CHROMA_PERSIST_DIRECTORY=./data/chroma_db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIEVAL_DOCS=10
RELEVANCE_THRESHOLD=0.5
```

## 📊 Usage

### 1. API Endpoints

#### Query Endpoint
```bash
curl -X POST "http://localhost:8000/api/query" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "How do I create a tensor in PyTorch?",
       "page_url": "https://pytorch.org/docs/stable/tensors.html",
       "include_citations": true
     }'
```

#### Health Check
```bash
curl "http://localhost:8000/api/health"
```

#### WebSocket Chat
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');
ws.send(JSON.stringify({
  type: 'query',
  data: {
    question: 'How to use torch.nn.Linear?',
    session_id: 'user123'
  }
}));
```

### 2. Chrome Extension

1. **Install the extension** following the setup instructions
2. **Navigate to PyTorch documentation** (e.g., pytorch.org/docs)
3. **Click the PyTorch RAG Assistant button** or use the extension popup
4. **Ask questions** about the current page or general PyTorch topics
5. **View citations** and source links in responses

### 3. Evaluation

```bash
# Run RAGAS evaluation
python -m backend.evaluation.ragas_eval

# Run performance tests
python -m backend.tests.test_pipeline
```

## 📈 Performance Metrics

The system tracks comprehensive metrics:

- **Faithfulness**: How factually consistent answers are with context
- **Answer Relevancy**: How relevant answers are to questions
- **Context Recall**: Whether all necessary context was retrieved
- **Answer Correctness**: Accuracy compared to ground truth
- **Context Precision**: Precision of retrieved context

## 🔍 API Documentation

### Query Request
```json
{
  "question": "string",
  "page_url": "string (optional)",
  "page_content": "string (optional)",
  "session_id": "string (optional)",
  "chat_history": "array (optional)",
  "include_citations": "boolean",
  "stream_response": "boolean"
}
```

### Query Response
```json
{
  "question": "string",
  "answer": "string",
  "citations": [
    {
      "id": "integer",
      "title": "string",
      "url": "string",
      "relevance_score": "float"
    }
  ],
  "confidence": "float",
  "pipeline_metrics": {
    "total_time": "float",
    "routing_time": "float",
    "retrieval_time": "float",
    "generation_time": "float"
  }
}
```

## 🧪 Testing

### Unit Tests
```bash
pytest backend/tests/
```

### Integration Tests
```bash
python backend/tests/test_pipeline.py
```

### RAGAS Evaluation
```bash
python backend/evaluation/ragas_eval.py
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t pytorch-rag-assistant .

# Run with Docker Compose
docker-compose up -d
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent documentation
- **LangChain** for RAG framework components
- **RAGAS** for evaluation metrics
- **ChromaDB** and **Neo4j** for data storage
- **Ollama** for local LLM inference



**Built with ❤️ for the PyTorch community**
