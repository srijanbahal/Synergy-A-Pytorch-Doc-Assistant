"""
Pydantic models for API request/response validation.
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from enum import Enum


class QueryType(str, Enum):
    """Query complexity types."""
    SIMPLE = "simple"
    COMPLEX = "complex"


class RouteType(str, Enum):
    """Routing types for query processing."""
    SIMPLE_VECTOR = "simple_vector"
    COMPLEX_HYBRID = "complex_hybrid"
    GRAPH_FOCUSED = "graph_focused"
    WEB_FALLBACK = "web_fallback"


class MessageType(str, Enum):
    """Message types for chat history."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# Request Models
class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., description="User's question", min_length=1, max_length=1000)
    page_url: Optional[HttpUrl] = Field(None, description="Current page URL")
    page_content: Optional[str] = Field(None, description="Current page HTML content")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    chat_history: Optional[List[Dict[str, Any]]] = Field(None, description="Previous conversation history")
    include_citations: bool = Field(True, description="Whether to include source citations")
    stream_response: bool = Field(False, description="Whether to stream the response")


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""
    session_id: str = Field(..., description="Session ID")
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    feedback_type: str = Field(..., description="Type of feedback: helpful, not_helpful, incorrect, etc.")
    feedback_text: Optional[str] = Field(None, description="Additional feedback text")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1-5")


class BatchQueryRequest(BaseModel):
    """Request model for batch queries."""
    queries: List[QueryRequest] = Field(..., description="List of queries to process")
    session_id: Optional[str] = Field(None, description="Session ID for batch processing")


# Response Models
class Citation(BaseModel):
    """Citation information."""
    id: int = Field(..., description="Citation ID")
    title: str = Field(..., description="Source title")
    url: str = Field(..., description="Source URL")
    module: Optional[str] = Field(None, description="PyTorch module")
    function: Optional[str] = Field(None, description="Function name")
    class_name: Optional[str] = Field(None, description="Class name")
    relevance_score: float = Field(..., description="Relevance score")


class ReflectionMetrics(BaseModel):
    """Self-RAG reflection metrics."""
    reflection_text: str = Field(..., description="Reflection analysis")
    scores: Dict[str, float] = Field(..., description="Quality scores")
    overall_score: float = Field(..., description="Overall quality score")


class PipelineMetrics(BaseModel):
    """Pipeline performance metrics."""
    total_time: float = Field(..., description="Total processing time in seconds")
    routing_time: float = Field(..., description="Routing time")
    retrieval_time: float = Field(..., description="Retrieval time")
    reranking_time: float = Field(..., description="Re-ranking time")
    generation_time: float = Field(..., description="Generation time")
    documents_retrieved: int = Field(..., description="Number of documents retrieved")
    documents_used: int = Field(..., description="Number of documents used in generation")


class RoutingInfo(BaseModel):
    """Routing decision information."""
    route_type: RouteType = Field(..., description="Chosen route type")
    confidence: float = Field(..., description="Confidence in routing decision")
    reasoning: str = Field(..., description="Reasoning for routing decision")
    complexity: QueryType = Field(..., description="Query complexity assessment")


class CorrectionAction(BaseModel):
    """Corrective action information."""
    action: str = Field(..., description="Corrective action taken")
    reason: str = Field(..., description="Reason for correction")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(..., description="Source citations")
    citation_text: str = Field(..., description="Formatted citation text")
    confidence: float = Field(..., description="Answer confidence score")
    reflection: ReflectionMetrics = Field(..., description="Self-RAG reflection")
    pipeline_metrics: PipelineMetrics = Field(..., description="Performance metrics")
    routing: RoutingInfo = Field(..., description="Routing information")
    correction: CorrectionAction = Field(..., description="Correction action")
    session_id: Optional[str] = Field(None, description="Session ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class StreamResponse(BaseModel):
    """Response model for streaming queries."""
    chunk: str = Field(..., description="Response chunk")
    is_complete: bool = Field(..., description="Whether this is the final chunk")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Overall health status")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")


class FeedbackResponse(BaseModel):
    """Feedback submission response."""
    success: bool = Field(..., description="Whether feedback was recorded")
    message: str = Field(..., description="Response message")
    feedback_id: Optional[str] = Field(None, description="Feedback ID")


class BatchQueryResponse(BaseModel):
    """Response model for batch queries."""
    results: List[QueryResponse] = Field(..., description="Individual query results")
    batch_metrics: Dict[str, Any] = Field(..., description="Batch processing metrics")
    session_id: Optional[str] = Field(None, description="Session ID")


class StatsResponse(BaseModel):
    """System statistics response."""
    vector_store: Dict[str, Any] = Field(..., description="Vector store statistics")
    graph_store: Dict[str, Any] = Field(..., description="Graph store statistics")
    components: Dict[str, str] = Field(..., description="Component information")


class ConversationSummary(BaseModel):
    """Conversation summary model."""
    session_id: str = Field(..., description="Session ID")
    message_count: int = Field(..., description="Total message count")
    user_messages: int = Field(..., description="User message count")
    assistant_messages: int = Field(..., description="Assistant message count")
    topics: List[str] = Field(..., description="Conversation topics")
    entities: List[str] = Field(..., description="Extracted entities")
    duration_seconds: float = Field(..., description="Conversation duration")
    created_at: float = Field(..., description="Creation timestamp")
    last_activity: float = Field(..., description="Last activity timestamp")


class SessionInfo(BaseModel):
    """Session information model."""
    session_id: str = Field(..., description="Session ID")
    exists: bool = Field(..., description="Whether session exists")
    message_count: int = Field(..., description="Message count")
    created_at: float = Field(..., description="Creation timestamp")
    last_activity: float = Field(..., description="Last activity timestamp")
    topics: List[str] = Field(..., description="Conversation topics")
    entities: List[str] = Field(..., description="Extracted entities")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# WebSocket Models
class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    session_id: Optional[str] = Field(None, description="Session ID")


class WebSocketQuery(BaseModel):
    """WebSocket query model."""
    question: str = Field(..., description="User question")
    page_url: Optional[str] = Field(None, description="Current page URL")
    page_content: Optional[str] = Field(None, description="Current page content")
    session_id: Optional[str] = Field(None, description="Session ID")


class WebSocketResponse(BaseModel):
    """WebSocket response model."""
    type: str = Field(..., description="Response type")
    data: Union[str, Dict[str, Any]] = Field(..., description="Response data")
    session_id: Optional[str] = Field(None, description="Session ID")
    is_complete: bool = Field(False, description="Whether response is complete")


# Configuration Models
class RateLimitInfo(BaseModel):
    """Rate limit information."""
    limit: int = Field(..., description="Rate limit per window")
    remaining: int = Field(..., description="Remaining requests")
    reset_time: float = Field(..., description="Reset timestamp")
    window_size: int = Field(..., description="Window size in seconds")
