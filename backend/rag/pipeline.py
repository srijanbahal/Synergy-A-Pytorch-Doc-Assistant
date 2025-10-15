"""
Main RAG pipeline orchestrating all components.
"""
import time
from typing import Dict, List, Optional, Any
from langchain.schema import Document
import structlog
from backend.rag.router import IntelligentRouter, RouteDecision
from backend.rag.retriever import HybridRetriever
from backend.rag.reranker import CRAGInspiredReranker
from backend.rag.generator import SelfRAGGenerator
from backend.stores.vector_store import PyTorchVectorStore
from backend.stores.graph_store import PyTorchGraphStore

logger = structlog.get_logger(__name__)


class PyTorchRAGPipeline:
    """Main RAG pipeline orchestrating all components."""
    
    def __init__(self):
        # Initialize components
        self.vector_store = PyTorchVectorStore()
        self.graph_store = PyTorchGraphStore()
        
        # Initialize RAG components
        self.router = IntelligentRouter()
        self.retriever = HybridRetriever(self.vector_store, self.graph_store)
        self.reranker = CRAGInspiredReranker()
        self.generator = SelfRAGGenerator()
        
        logger.info("PyTorch RAG pipeline initialized")
    
    def process_query(self, question: str, page_context: Optional[Dict] = None,
                     session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a query through the complete RAG pipeline."""
        start_time = time.time()
        
        try:
            # Step 1: Route the query
            routing_start = time.time()
            route_decision = self.router.determine_route(question, page_context)
            route_config = self.router.get_route_config(route_decision.route_type)
            routing_time = time.time() - routing_start
            
            logger.info(f"Query routed as: {route_decision.route_type} (confidence: {route_decision.confidence:.2f})")
            
            # Step 2: Retrieve relevant documents
            retrieval_start = time.time()
            documents = self.retriever.get_relevant_documents(
                query=question,
                route_config=route_config,
                page_context=page_context
            )
            retrieval_time = time.time() - retrieval_start
            
            logger.info(f"Retrieved {len(documents)} documents")
            
            # Step 3: Re-rank and filter documents
            reranking_start = time.time()
            if route_config.get('reranking', False):
                documents = self.reranker.rerank_documents(question, documents)
            
            # Assess retrieval quality and apply corrections if needed
            corrective_action = self.reranker.get_corrective_action(question, documents)
            documents = self.reranker.apply_correction(question, documents, corrective_action)
            
            reranking_time = time.time() - reranking_start
            
            logger.info(f"Re-ranked to {len(documents)} documents")
            
            # Step 4: Generate answer
            generation_start = time.time()
            response = self.generator.generate_with_retry(question, documents)
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            
            # Compile pipeline results
            pipeline_result = {
                "question": question,
                "answer": response["answer"],
                "citations": response["citations"],
                "citation_text": response["citation_text"],
                "confidence": response["confidence"],
                "reflection": response["reflection"],
                "pipeline_metrics": {
                    "total_time": total_time,
                    "routing_time": routing_time,
                    "retrieval_time": retrieval_time,
                    "reranking_time": reranking_time,
                    "generation_time": generation_time,
                    "documents_retrieved": len(documents),
                    "documents_used": response["context_used"]
                },
                "routing": {
                    "route_type": route_decision.route_type,
                    "confidence": route_decision.confidence,
                    "reasoning": route_decision.reasoning,
                    "complexity": route_decision.query_complexity
                },
                "correction": corrective_action,
                "session_id": session_id
            }
            
            logger.info(f"Query processed successfully in {total_time:.2f}s")
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                "question": question,
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "citations": [],
                "citation_text": "",
                "confidence": 0.0,
                "error": str(e),
                "pipeline_metrics": {
                    "total_time": time.time() - start_time,
                    "error": True
                }
            }
    
    def process_batch_queries(self, queries: List[Dict]) -> List[Dict[str, Any]]:
        """Process multiple queries in batch."""
        results = []
        
        for query_data in queries:
            question = query_data.get("question", "")
            page_context = query_data.get("page_context")
            session_id = query_data.get("session_id")
            
            result = self.process_query(question, page_context, session_id)
            results.append(result)
        
        logger.info(f"Processed {len(queries)} queries in batch")
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline components."""
        try:
            vector_stats = self.vector_store.get_stats()
            graph_stats = self.graph_store.get_stats()
            
            return {
                "vector_store": vector_stats,
                "graph_store": graph_stats,
                "components": {
                    "router": "IntelligentRouter",
                    "retriever": "HybridRetriever",
                    "reranker": "CRAGInspiredReranker",
                    "generator": "SelfRAGGenerator"
                }
            }
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {e}")
            return {"error": str(e)}
    
    def reset_pipeline(self) -> None:
        """Reset the pipeline components."""
        try:
            self.vector_store.reset()
            self.graph_store.clear_all()
            logger.info("Pipeline reset successfully")
        except Exception as e:
            logger.error(f"Error resetting pipeline: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all pipeline components."""
        health_status = {
            "overall": "healthy",
            "components": {},
            "timestamp": time.time()
        }
        
        # Check vector store
        try:
            vector_stats = self.vector_store.get_stats()
            health_status["components"]["vector_store"] = {
                "status": "healthy",
                "documents_count": vector_stats.get("documents_count", 0),
                "summaries_count": vector_stats.get("summaries_count", 0)
            }
        except Exception as e:
            health_status["components"]["vector_store"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall"] = "degraded"
        
        # Check graph store
        try:
            graph_stats = self.graph_store.get_stats()
            health_status["components"]["graph_store"] = {
                "status": "healthy",
                "nodes": graph_stats.get("nodes", {}),
                "relationships": graph_stats.get("relationships", {})
            }
        except Exception as e:
            health_status["components"]["graph_store"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall"] = "degraded"
        
        # Check LLM (simple test)
        try:
            test_response = self.generator.llm.invoke({"question": "test"})
            health_status["components"]["llm"] = {
                "status": "healthy",
                "response_length": len(str(test_response))
            }
        except Exception as e:
            health_status["components"]["llm"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall"] = "degraded"
        
        return health_status
    
    def explain_route_decision(self, question: str, page_context: Optional[Dict] = None) -> Dict:
        """Explain why a particular route was chosen for a query."""
        route_decision = self.router.determine_route(question, page_context)
        route_config = self.router.get_route_config(route_decision.route_type)
        
        return {
            "question": question,
            "route_decision": {
                "route_type": route_decision.route_type,
                "confidence": route_decision.confidence,
                "reasoning": route_decision.reasoning,
                "complexity": route_decision.query_complexity
            },
            "route_config": route_config,
            "explanation": {
                "why_this_route": f"Query was classified as '{route_decision.query_complexity}' complexity",
                "processing_steps": self._get_route_processing_steps(route_decision.route_type),
                "expected_performance": self._get_expected_performance(route_decision.route_type)
            }
        }
    
    def _get_route_processing_steps(self, route_type: str) -> List[str]:
        """Get the processing steps for a given route type."""
        steps = {
            "simple_vector": [
                "1. Direct vector similarity search",
                "2. Basic relevance filtering",
                "3. Answer generation with citations"
            ],
            "complex_hybrid": [
                "1. Multi-query generation",
                "2. Parallel vector + graph search",
                "3. Reciprocal Rank Fusion re-ranking",
                "4. Cross-encoder relevance scoring",
                "5. Answer generation with Self-RAG reflection"
            ],
            "graph_focused": [
                "1. Entity extraction from query",
                "2. Graph traversal for related entities",
                "3. Vector search for context",
                "4. Relationship-based answer generation"
            ],
            "web_fallback": [
                "1. Basic vector search",
                "2. Web search for recent information",
                "3. Combined answer generation"
            ]
        }
        
        return steps.get(route_type, ["Unknown route type"])


# Global pipeline instance
_pipeline_instance = None


def get_pipeline() -> PyTorchRAGPipeline:
    """Get the global pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = PyTorchRAGPipeline()
    return _pipeline_instance


def main():
    """Test the RAG pipeline."""
    pipeline = PyTorchRAGPipeline()
    
    # Test queries
    test_queries = [
        "How do I create a tensor in PyTorch?",
        "What's the difference between nn.Linear and nn.Conv2d?",
        "How to implement a neural network?",
    ]
    
    for question in test_queries:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        
        result = pipeline.process_query(question)
        
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Citations: {len(result['citations'])}")
        print(f"Processing time: {result['pipeline_metrics']['total_time']:.2f}s")
        print(f"Route: {result['routing']['route_type']}")
    
    # Test health check
    print(f"\n{'='*50}")
    print("Health Check:")
    health = pipeline.health_check()
    print(f"Overall: {health['overall']}")
    for component, status in health['components'].items():
        print(f"{component}: {status['status']}")


if __name__ == "__main__":
    main()
