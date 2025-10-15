@echo off
echo 🚀 Starting PyTorch RAG Assistant...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker and try again.
    pause
    exit /b 1
)

REM Create necessary directories
echo 📁 Creating directories...
if not exist "data\chroma_db" mkdir "data\chroma_db"
if not exist "data\cache" mkdir "data\cache"
if not exist "logs" mkdir "logs"
if not exist "evaluation_cache" mkdir "evaluation_cache"

REM Start services with Docker Compose
echo 🐳 Starting services with Docker Compose...
docker-compose up -d

REM Wait for services to be ready
echo ⏳ Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Check if Neo4j is ready
echo 🔍 Checking Neo4j connection...
set /a attempt=1
:neo4j_check
curl -s http://localhost:7474 >nul 2>&1
if errorlevel 1 (
    if %attempt% LEQ 30 (
        echo ⏳ Waiting for Neo4j... (attempt %attempt%/30)
        timeout /t 2 /nobreak >nul
        set /a attempt+=1
        goto neo4j_check
    ) else (
        echo ❌ Neo4j failed to start within expected time
        pause
        exit /b 1
    )
) else (
    echo ✅ Neo4j is ready!
)

REM Check if backend API is ready
echo 🔍 Checking backend API...
set /a attempt=1
:api_check
curl -s http://localhost:8000/api/health >nul 2>&1
if errorlevel 1 (
    if %attempt% LEQ 30 (
        echo ⏳ Waiting for backend API... (attempt %attempt%/30)
        timeout /t 2 /nobreak >nul
        set /a attempt+=1
        goto api_check
    ) else (
        echo ❌ Backend API failed to start within expected time
        pause
        exit /b 1
    )
) else (
    echo ✅ Backend API is ready!
)

echo.
echo 🎉 PyTorch RAG Assistant is ready!
echo.
echo 📊 Services Status:
echo   - Neo4j: http://localhost:7474 (neo4j/password)
echo   - Backend API: http://localhost:8000
echo   - API Docs: http://localhost:8000/docs
echo   - Redis: localhost:6379
echo.
echo 🔧 Next Steps:
echo   1. Install Chrome Extension from chrome_extension\ directory
echo   2. Navigate to PyTorch documentation (e.g., pytorch.org/docs)
echo   3. Use the PyTorch RAG Assistant!
echo.
echo 📚 Useful Commands:
echo   - View logs: docker-compose logs -f
echo   - Stop services: docker-compose down
echo   - Restart: docker-compose restart
echo   - Run evaluation: docker-compose exec backend python -m backend.evaluation.ragas_eval
echo.
echo Happy coding! 🚀
pause
