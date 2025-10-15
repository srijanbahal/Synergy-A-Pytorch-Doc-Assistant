#!/bin/bash

# PyTorch RAG Assistant Startup Script

echo "🚀 Starting PyTorch RAG Assistant..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama is not running. Starting Ollama..."
    echo "Please make sure Ollama is installed and running on port 11434"
    echo "You can start it with: ollama serve"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/chroma_db
mkdir -p data/cache
mkdir -p logs
mkdir -p evaluation_cache

# Start services with Docker Compose
echo "🐳 Starting services with Docker Compose..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if Neo4j is ready
echo "🔍 Checking Neo4j connection..."
max_attempts=30
attempt=1
while [ $attempt -le $max_attempts ]; do
    if curl -s http://localhost:7474 > /dev/null 2>&1; then
        echo "✅ Neo4j is ready!"
        break
    fi
    echo "⏳ Waiting for Neo4j... (attempt $attempt/$max_attempts)"
    sleep 2
    ((attempt++))
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ Neo4j failed to start within expected time"
    exit 1
fi

# Check if backend API is ready
echo "🔍 Checking backend API..."
max_attempts=30
attempt=1
while [ $attempt -le $max_attempts ]; do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "✅ Backend API is ready!"
        break
    fi
    echo "⏳ Waiting for backend API... (attempt $attempt/$max_attempts)"
    sleep 2
    ((attempt++))
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ Backend API failed to start within expected time"
    exit 1
fi

# Initialize databases (optional)
echo "🗄️  Initializing databases..."
docker-compose exec backend python -c "
from backend.stores.graph_store import Neo4jGraphStore
from backend.stores.vector_store import ChromaVectorStore

# Initialize Neo4j schema
graph_store = Neo4jGraphStore()
graph_store.initialize_schema()
print('✅ Neo4j schema initialized')

# Initialize ChromaDB
vector_store = ChromaVectorStore()
print('✅ ChromaDB initialized')
" 2>/dev/null || echo "⚠️  Database initialization skipped (may need manual setup)"

echo ""
echo "🎉 PyTorch RAG Assistant is ready!"
echo ""
echo "📊 Services Status:"
echo "  - Neo4j: http://localhost:7474 (neo4j/password)"
echo "  - Backend API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Redis: localhost:6379"
echo ""
echo "🔧 Next Steps:"
echo "  1. Install Chrome Extension from chrome_extension/ directory"
echo "  2. Navigate to PyTorch documentation (e.g., pytorch.org/docs)"
echo "  3. Use the PyTorch RAG Assistant!"
echo ""
echo "📚 Useful Commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop services: docker-compose down"
echo "  - Restart: docker-compose restart"
echo "  - Run evaluation: docker-compose exec backend python -m backend.evaluation.ragas_eval"
echo ""
echo "Happy coding! 🚀"
