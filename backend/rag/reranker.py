"""
Re-ranking and relevance filtering for retrieved documents.
"""
import re
from typing import Dict, List, Optional, Tuple
from langchain.schema import Document
import structlog
from backend.config import settings

logger = structlog.get_logger(__name__)


class RelevanceReranker:
    """Re-ranker for filtering and improving document relevance."""
    
    def __init__(self):
        self.relevance_threshold = settings.relevance_threshold
        
        # Initialize cross-encoder for re-ranking (optional)
        self.cross_encoder = None
        self._initialize_cross_encoder()
        
        logger.info("Relevance reranker initialized")
    
    def _initialize_cross_encoder(self):
        """Initialize cross-encoder model for re-ranking."""
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Cross-encoder initialized for re-ranking")
        except Exception as e:
            logger.warning(f"Failed to initialize cross-encoder: {e}")
    
    def calculate_relevance_score(self, query: str, document: Document) -> float:
        """Calculate relevance score between query and document."""
        doc_content = document.page_content.lower()
        query_terms = query.lower().split()
        
        # Basic term frequency scoring
        term_scores = []
        for term in query_terms:
            if len(term) > 2:  # Skip short terms
                count = doc_content.count(term)
                if count > 0:
                    # Normalize by document length
                    normalized_score = count / len(doc_content.split())
                    term_scores.append(normalized_score)
        
        # Average term score
        if term_scores:
            basic_score = sum(term_scores) / len(term_scores)
        else:
            basic_score = 0.0
        
        # Boost score for exact phrase matches
        if query.lower() in doc_content:
            basic_score += 0.3
        
        # Boost score for metadata matches
        metadata = document.metadata
        if metadata:
            # Check if query terms appear in metadata
            metadata_text = " ".join([
                str(metadata.get('title', '')),
                str(metadata.get('function', '')),
                str(metadata.get('class', '')),
                str(metadata.get('module', ''))
            ]).lower()
            
            metadata_matches = sum(1 for term in query_terms if term in metadata_text)
            if metadata_matches > 0:
                basic_score += 0.2 * (metadata_matches / len(query_terms))
        
        return min(basic_score, 1.0)
    
    def cross_encoder_rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Re-rank documents using cross-encoder."""
        if not self.cross_encoder or not documents:
            return [(doc, 0.0) for doc in documents]
        
        try:
            # Prepare pairs for cross-encoder
            pairs = [(query, doc.page_content) for doc in documents]
            
            # Get relevance scores
            scores = self.cross_encoder.predict(pairs)
            
            # Combine documents with scores
            scored_docs = list(zip(documents, scores))
            
            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Cross-encoder re-ranked {len(documents)} documents")
            return scored_docs
            
        except Exception as e:
            logger.error(f"Error in cross-encoder re-ranking: {e}")
            return [(doc, 0.0) for doc in documents]
    
    def filter_by_relevance(self, documents: List[Document], query: str,
                          threshold: Optional[float] = None) -> List[Document]:
        """Filter documents by relevance threshold."""
        if threshold is None:
            threshold = self.relevance_threshold
        
        filtered_docs = []
        
        for doc in documents:
            score = self.calculate_relevance_score(query, doc)
            
            # Add score to metadata
            doc.metadata['relevance_score'] = score
            
            if score >= threshold:
                filtered_docs.append(doc)
        
        logger.debug(f"Filtered {len(documents)} -> {len(filtered_docs)} documents (threshold: {threshold})")
        return filtered_docs
    
    def rerank_documents(self, query: str, documents: List[Document],
                        use_cross_encoder: bool = True) -> List[Document]:
        """Re-rank documents using multiple methods."""
        if not documents:
            return []
        
        try:
            # First, filter by basic relevance
            filtered_docs = self.filter_by_relevance(documents, query)
            
            if not filtered_docs:
                logger.warning("No documents passed relevance filter")
                return []
            
            # Use cross-encoder if available
            if use_cross_encoder and self.cross_encoder:
                scored_docs = self.cross_encoder_rerank(query, filtered_docs)
                reranked_docs = [doc for doc, score in scored_docs]
            else:
                # Sort by relevance score
                reranked_docs = sorted(
                    filtered_docs,
                    key=lambda x: x.metadata.get('relevance_score', 0),
                    reverse=True
                )
            
            logger.info(f"Re-ranked {len(documents)} -> {len(reranked_docs)} documents")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error re-ranking documents: {e}")
            return documents
    
    def evaluate_retrieval_quality(self, query: str, documents: List[Document]) -> Dict:
        """Evaluate the quality of retrieved documents."""
        if not documents:
            return {
                "total_documents": 0,
                "avg_relevance_score": 0.0,
                "high_quality_docs": 0,
                "quality_assessment": "poor"
            }
        
        # Calculate average relevance score
        scores = [doc.metadata.get('relevance_score', 0) for doc in documents]
        avg_score = sum(scores) / len(scores)
        
        # Count high-quality documents
        high_quality_threshold = 0.7
        high_quality_docs = sum(1 for score in scores if score >= high_quality_threshold)
        
        # Determine overall quality
        if avg_score >= 0.8 and high_quality_docs >= len(documents) * 0.8:
            quality_assessment = "excellent"
        elif avg_score >= 0.6 and high_quality_docs >= len(documents) * 0.6:
            quality_assessment = "good"
        elif avg_score >= 0.4 and high_quality_docs >= len(documents) * 0.4:
            quality_assessment = "fair"
        else:
            quality_assessment = "poor"
        
        return {
            "total_documents": len(documents),
            "avg_relevance_score": avg_score,
            "high_quality_docs": high_quality_docs,
            "quality_assessment": quality_assessment
        }
    
    def should_trigger_fallback(self, query: str, documents: List[Document]) -> bool:
        """Determine if fallback search should be triggered."""
        quality = self.evaluate_retrieval_quality(query, documents)
        
        # Trigger fallback if quality is poor
        return quality["quality_assessment"] == "poor"
    
    def get_top_documents(self, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Get top-k documents by relevance score."""
        if not documents:
            return []
        
        # Sort by relevance score
        sorted_docs = sorted(
            documents,
            key=lambda x: x.metadata.get('relevance_score', 0),
            reverse=True
        )
        
        return sorted_docs[:top_k]


class CRAGInspiredReranker(RelevanceReranker):
    """CRAG-inspired reranker with corrective mechanisms."""
    
    def __init__(self):
        super().__init__()
        self.correction_threshold = 0.3  # Lower threshold for triggering corrections
        logger.info("CRAG-inspired reranker initialized")
    
    def assess_retrieval_quality(self, query: str, documents: List[Document]) -> str:
        """Assess retrieval quality and determine corrective action."""
        if not documents:
            return "irrelevant"
        
        # Calculate average relevance
        scores = [self.calculate_relevance_score(query, doc) for doc in documents]
        avg_relevance = sum(scores) / len(scores)
        
        # Count highly relevant documents
        high_relevance_count = sum(1 for score in scores if score >= 0.7)
        high_relevance_ratio = high_relevance_count / len(documents)
        
        # Determine quality level
        if avg_relevance >= 0.7 and high_relevance_ratio >= 0.6:
            return "correct"
        elif avg_relevance >= 0.4 and high_relevance_ratio >= 0.3:
            return "incorrect"
        else:
            return "irrelevant"
    
    def get_corrective_action(self, query: str, documents: List[Document]) -> Dict:
        """Determine corrective action based on retrieval quality."""
        quality = self.assess_retrieval_quality(query, documents)
        
        if quality == "correct":
            return {
                "action": "proceed",
                "reason": "Retrieved documents are relevant and sufficient"
            }
        elif quality == "incorrect":
            return {
                "action": "web_search",
                "reason": "Documents are partially relevant, web search may help"
            }
        else:  # irrelevant
            return {
                "action": "broaden_search",
                "reason": "Documents are not relevant, need broader search"
            }
    
    def apply_correction(self, query: str, documents: List[Document], 
                        corrective_action: Dict) -> List[Document]:
        """Apply corrective action to improve retrieval."""
        action = corrective_action["action"]
        
        if action == "proceed":
            return documents
        
        elif action == "web_search":
            # For now, just return the original documents
            # In a full implementation, this would trigger web search
            logger.info("Web search correction triggered")
            return documents
        
        elif action == "broaden_search":
            # Return a subset of documents and indicate need for broader search
            logger.info("Broaden search correction triggered")
            return documents[:3] if documents else []
        
        return documents


def main():
    """Test the reranker."""
    reranker = RelevanceReranker()
    
    # Create test documents
    test_docs = [
        Document(
            page_content="torch.tensor creates a tensor from data",
            metadata={"function": "tensor", "module": "torch"}
        ),
        Document(
            page_content="torch.nn.Linear applies a linear transformation",
            metadata={"class": "Linear", "module": "torch.nn"}
        ),
        Document(
            page_content="Python programming language basics",
            metadata={"topic": "python"}
        )
    ]
    
    query = "how to create a tensor"
    
    # Test relevance calculation
    for i, doc in enumerate(test_docs):
        score = reranker.calculate_relevance_score(query, doc)
        print(f"Doc {i+1} relevance score: {score:.3f}")
    
    # Test filtering and re-ranking
    filtered_docs = reranker.filter_by_relevance(test_docs, query, threshold=0.1)
    print(f"Filtered documents: {len(filtered_docs)}")
    
    reranked_docs = reranker.rerank_documents(query, test_docs)
    print(f"Re-ranked documents: {len(reranked_docs)}")
    
    # Test quality evaluation
    quality = reranker.evaluate_retrieval_quality(query, test_docs)
    print(f"Quality assessment: {quality}")


if __name__ == "__main__":
    main()
