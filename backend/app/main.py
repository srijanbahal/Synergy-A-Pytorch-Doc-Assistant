"""
FastAPI main application with all endpoints and WebSocket support.
"""
import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import structlog
from backend.config import settings
from backend.app.models import (
    QueryRequest, QueryResponse, FeedbackRequest, FeedbackResponse,
    BatchQueryRequest, BatchQueryResponse, HealthResponse, StatsResponse,
    ConversationSummary, SessionInfo, ErrorResponse, WebSocketMessage,
    WebSocketQuery, WebSocketResponse
)
from backend.app.middleware import setup_middleware
from backend.rag.pipeline import get_pipeline
from backend.context.page_analyzer import PageAnalyzer
from backend.memory.chat_history import get_conversation_memory

logger = structlog.get_logger(__name__)

# Global instances
pipeline = None
page_analyzer = None
conversation_memory = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global pipeline, page_analyzer, conversation_memory
    
    # Startup
    logger.info("Starting PyTorch RAG Assistant API")
    
    try:
        # Initialize components
        pipeline = get_pipeline()
        page_analyzer = PageAnalyzer()
        conversation_memory = get_conversation_memory()
        
        logger.info("All components initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down PyTorch RAG Assistant API")


# Create FastAPI app
app = FastAPI(
    title="PyTorch RAG Assistant API",
    description="Advanced RAG system for PyTorch documentation assistance",
    version="1.0.0",
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)


# Dependency to get pipeline
def get_rag_pipeline():
    """Get the RAG pipeline instance."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return pipeline


# Health check endpoint
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        health_info = pipeline.health_check()
        return HealthResponse(**health_info)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# Main query endpoint
@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, rag_pipeline=Depends(get_rag_pipeline)):
    """Main RAG query endpoint."""
    try:
        start_time = time.time()
        
        # Analyze page content if provided
        page_context = None
        if request.page_content:
            page_context = page_analyzer.analyze_page(
                request.page_content, 
                str(request.page_url) if request.page_url else ""
            )
        
        # Add user message to conversation history
        if request.session_id:
            conversation_memory.add_message(request.session_id, {
                "type": "user",
                "content": request.question,
                "page_url": str(request.page_url) if request.page_url else None,
                "page_context": page_context
            })
        
        # Process query through RAG pipeline
        result = rag_pipeline.process_query(
            question=request.question,
            page_context=page_context,
            session_id=request.session_id
        )
        
        # Add assistant response to conversation history
        if request.session_id:
            conversation_memory.add_message(request.session_id, {
                "type": "assistant",
                "content": result["answer"],
                "citations": result.get("citations", []),
                "confidence": result.get("confidence", 0.0)
            })
        
        # Convert to response model
        response = QueryResponse(**result)
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f}s with confidence {result['confidence']:.2f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


# Batch query endpoint
@app.post("/api/query/batch", response_model=BatchQueryResponse)
async def batch_query_endpoint(request: BatchQueryRequest, rag_pipeline=Depends(get_rag_pipeline)):
    """Batch query processing endpoint."""
    try:
        start_time = time.time()
        
        # Process all queries
        results = []
        for query_req in request.queries:
            # Analyze page content if provided
            page_context = None
            if query_req.page_content:
                page_context = page_analyzer.analyze_page(
                    query_req.page_content,
                    str(query_req.page_url) if query_req.page_url else ""
                )
            
            # Process query
            result = rag_pipeline.process_query(
                question=query_req.question,
                page_context=page_context,
                session_id=request.session_id or query_req.session_id
            )
            
            results.append(QueryResponse(**result))
        
        batch_metrics = {
            "total_queries": len(request.queries),
            "processing_time": time.time() - start_time,
            "average_confidence": sum(r.confidence for r in results) / len(results) if results else 0
        }
        
        return BatchQueryResponse(
            results=results,
            batch_metrics=batch_metrics,
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Batch query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


# Feedback endpoint
@app.post("/api/feedback", response_model=FeedbackResponse)
async def feedback_endpoint(request: FeedbackRequest):
    """User feedback endpoint for RAGAS evaluation."""
    try:
        # Store feedback (in production, this would go to a database)
        feedback_id = f"feedback_{int(time.time())}"
        
        # Log feedback for evaluation
        logger.info(
            "User feedback received",
            feedback_id=feedback_id,
            session_id=request.session_id,
            feedback_type=request.feedback_type,
            rating=request.rating,
            query=request.query[:100] + "..." if len(request.query) > 100 else request.query
        )
        
        return FeedbackResponse(
            success=True,
            message="Feedback recorded successfully",
            feedback_id=feedback_id
        )
        
    except Exception as e:
        logger.error(f"Feedback recording failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")


# Stats endpoint
@app.get("/api/stats", response_model=StatsResponse)
async def stats_endpoint(rag_pipeline=Depends(get_rag_pipeline)):
    """System statistics endpoint."""
    try:
        stats = rag_pipeline.get_pipeline_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve stats")


# Conversation management endpoints
@app.get("/api/conversation/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get conversation session information."""
    try:
        info = conversation_memory.get_session_info(session_id)
        return SessionInfo(session_id=session_id, **info)
    except Exception as e:
        logger.error(f"Session info retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session info")


@app.get("/api/conversation/{session_id}/summary", response_model=ConversationSummary)
async def get_conversation_summary(session_id: str):
    """Get conversation summary."""
    try:
        summary = conversation_memory.summarize_conversation(session_id)
        return ConversationSummary(**summary)
    except Exception as e:
        logger.error(f"Conversation summary retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation summary")


@app.delete("/api/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history."""
    try:
        conversation_memory.clear_conversation(session_id)
        return {"message": "Conversation cleared successfully"}
    except Exception as e:
        logger.error(f"Conversation clearing failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear conversation")


# WebSocket endpoint for real-time chat
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    
    session_id = None
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "query":
                query_data = WebSocketQuery(**message["data"])
                session_id = query_data.session_id or f"ws_{int(time.time())}"
                
                # Analyze page content if provided
                page_context = None
                if query_data.page_content:
                    page_context = page_analyzer.analyze_page(
                        query_data.page_content,
                        query_data.page_url or ""
                    )
                
                # Add user message to conversation
                conversation_memory.add_message(session_id, {
                    "type": "user",
                    "content": query_data.question,
                    "page_url": query_data.page_url,
                    "page_context": page_context
                })
                
                # Process query
                result = pipeline.process_query(
                    question=query_data.question,
                    page_context=page_context,
                    session_id=session_id
                )
                
                # Add assistant response to conversation
                conversation_memory.add_message(session_id, {
                    "type": "assistant",
                    "content": result["answer"],
                    "citations": result.get("citations", []),
                    "confidence": result.get("confidence", 0.0)
                })
                
                # Send response
                response = WebSocketResponse(
                    type="response",
                    data=result,
                    session_id=session_id,
                    is_complete=True
                )
                
                await websocket.send_text(response.json())
            
            elif message["type"] == "ping":
                # Respond to ping
                pong_response = WebSocketResponse(
                    type="pong",
                    data={"timestamp": time.time()},
                    session_id=session_id
                )
                await websocket.send_text(pong_response.json())
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            error_response = WebSocketResponse(
                type="error",
                data={"error": str(e)},
                session_id=session_id
            )
            await websocket.send_text(error_response.json())
        except:
            pass


# Streaming response endpoint
@app.post("/api/query/stream")
async def stream_query_endpoint(request: QueryRequest, rag_pipeline=Depends(get_rag_pipeline)):
    """Streaming query endpoint."""
    try:
        def generate_response():
            try:
                # Analyze page content if provided
                page_context = None
                if request.page_content:
                    page_context = page_analyzer.analyze_page(
                        request.page_content,
                        str(request.page_url) if request.page_url else ""
                    )
                
                # Process query
                result = rag_pipeline.process_query(
                    question=request.question,
                    page_context=page_context,
                    session_id=request.session_id
                )
                
                # Stream the answer word by word
                answer = result["answer"]
                words = answer.split()
                
                for i, word in enumerate(words):
                    chunk_data = {
                        "chunk": word + " ",
                        "is_complete": i == len(words) - 1,
                        "metadata": {
                            "confidence": result["confidence"],
                            "citations": result.get("citations", []) if i == len(words) - 1 else []
                        }
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    time.sleep(0.05)  # Small delay for streaming effect
                
            except Exception as e:
                error_data = {
                    "chunk": f"Error: {str(e)}",
                    "is_complete": True,
                    "error": True
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming query failed: {e}")
        raise HTTPException(status_code=500, detail="Streaming failed")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "PyTorch RAG Assistant API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health"
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "timestamp": time.time()
        }
    )


@app.exception_handler(422)
async def validation_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Invalid request data",
            "details": exc.errors(),
            "timestamp": time.time()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level="info"
    )
