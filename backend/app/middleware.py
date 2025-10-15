"""
Middleware for CORS, rate limiting, and error handling.
"""
import time
from typing import Dict, Any
from fastapi import Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from collections import defaultdict, deque
import structlog
from backend.config import settings

logger = structlog.get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, calls: int = 60, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = defaultdict(lambda: deque())
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting."""
        # Get client identifier (IP address)
        client_ip = request.client.host
        
        # Clean old requests
        now = time.time()
        while self.clients[client_ip] and self.clients[client_ip][0] <= now - self.period:
            self.clients[client_ip].popleft()
        
        # Check rate limit
        if len(self.clients[client_ip]) >= self.calls:
            logger.warning(f"Rate limit exceeded for client {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.calls} requests per {self.period} seconds",
                    "retry_after": self.period
                }
            )
        
        # Add current request
        self.clients[client_ip].append(now)
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(self.calls - len(self.clients[client_ip]))
        response.headers["X-RateLimit-Reset"] = str(int(now + self.period))
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware."""
    
    async def dispatch(self, request: Request, call_next):
        """Log requests and responses."""
        start_time = time.time()
        
        # Log request
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host,
            user_agent=request.headers.get("user-agent", "unknown")
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=process_time
            )
            
            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                "Request failed",
                method=request.method,
                url=str(request.url),
                error=str(e),
                process_time=process_time
            )
            
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware."""
    
    async def dispatch(self, request: Request, call_next):
        """Handle errors globally."""
        try:
            response = await call_next(request)
            return response
            
        except HTTPException:
            # Let FastAPI handle HTTP exceptions
            raise
            
        except Exception as e:
            logger.error(
                "Unhandled exception",
                method=request.method,
                url=str(request.url),
                error=str(e),
                exc_info=True
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "timestamp": time.time()
                }
            )


def setup_cors(app):
    """Setup CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


def setup_middleware(app):
    """Setup all middleware."""
    # Add custom middleware
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        calls=settings.rate_limit_per_minute,
        period=60
    )
    
    # Setup CORS
    setup_cors(app)
    
    logger.info("Middleware setup complete")
