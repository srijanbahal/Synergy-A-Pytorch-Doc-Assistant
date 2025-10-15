"""
Hybrid retrieval system combining vector and graph search with RRF re-ranking.
"""
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from langchain.schema import Document
from langchain.load import dumps, loads
import structlog
from backend.stores.vector_store import PyTorchVectorStore
from backend.stores.graph_store import PyTorchGraphStore
from backend.rag.query_transformer import QueryTransformer

logger = structlog.get_logger(__name__)


class HybridRetriever:
    """Hybrid retriever combining vector search and graph traversal."""
    
    def __init__(self, vector_store: PyTorchVectorStore, graph_store: PyTorchGraphStore):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.query_transformer = QueryTransformer()
        
        logger.info("Hybrid retriever initialized")
    
    def reciprocal_rank_fusion(self, results_list: List[List[Document]], k: int = 60) -> List[Tuple[Document, float]]:
        """Apply Reciprocal Rank Fusion (RRF) to combine multiple ranked lists."""
        fused_scores = {}
        
        for results in results_list:
            for rank, doc in enumerate(results):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                
                # RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)
        
        # Sort by fused scores
        reranked_results = [
            (loads(doc_str), score)
            for doc_str, score in sorted(
                fused_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
        ]
        
        logger.debug(f"RRF applied to {len(results_list)} result lists, {len(reranked_results)} unique documents")
        return reranked_results
    
    def vector_search(self, query: str, k: int = 10, 
                     collection: str = "documents") -> List[Document]:
        """Perform vector similarity search."""
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                collection_name=collection
            )
            
            logger.debug(f"Vector search returned {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def graph_search(self, query: str, max_depth: int = 2) -> List[Document]:
        """Perform graph-based search by finding related entities."""
        try:
            # Search for entities mentioned in the query
            entities = self.graph_store.search_entities(query)
            
            if not entities:
                logger.debug("No entities found for graph search")
                return []
            
            all_related = []
            
            # For each entity, find related entities
            for entity in entities[:5]:  # Limit to top 5 entities
                entity_name = entity['name']
                entity_type = entity['entity_type']
                
                # Find related entities
                related = self.graph_store.find_related_entities(
                    entity_name=entity_name,
                    entity_type=entity_type,
                    max_depth=max_depth
                )
                
                # Convert to Document objects
                for rel in related:
                    content = f"{rel['entity_type']}: {rel['entity_name']}"
                    if rel.get('description'):
                        content += f"\nDescription: {rel['description']}"
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            'entity_type': rel['entity_type'],
                            'entity_name': rel['entity_name'],
                            'distance': rel['distance'],
                            'source': 'graph_search',
                            'original_entity': entity_name
                        }
                    )
                    all_related.append(doc)
            
            # Remove duplicates and limit results
            unique_docs = []
            seen_content = set()
            
            for doc in all_related:
                if doc.page_content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.page_content)
                    
                    if len(unique_docs) >= 10:
                        break
            
            logger.debug(f"Graph search returned {len(unique_docs)} documents")
            return unique_docs
            
        except Exception as e:
            logger.error(f"Error in graph search: {e}")
            return []
    
    def multi_query_search(self, query: str, k: int = 10) -> List[Document]:
        """Perform multi-query search with RRF."""
        try:
            # Generate alternative queries
            alternative_queries = self.query_transformer.generate_multi_queries(query)
            
            # Search with each query
            all_results = []
            for alt_query in alternative_queries:
                results = self.vector_search(alt_query, k)
                all_results.append(results)
            
            # Apply RRF
            reranked_results = self.reciprocal_rank_fusion(all_results)
            
            # Return documents without scores
            documents = [doc for doc, score in reranked_results]
            
            logger.debug(f"Multi-query search returned {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error in multi-query search: {e}")
            return []
    
    def rag_fusion_search(self, query: str, k: int = 10) -> List[Document]:
        """Perform RAG-Fusion search with enhanced re-ranking."""
        try:
            # Generate RAG-Fusion queries
            fusion_queries = self.query_transformer.generate_rag_fusion_queries(query)
            
            # Search with each query
            all_results = []
            for fusion_query in fusion_queries:
                results = self.vector_search(fusion_query, k)
                all_results.append(results)
            
            # Apply RRF with higher k for better fusion
            reranked_results = self.reciprocal_rank_fusion(all_results, k=100)
            
            # Return documents without scores
            documents = [doc for doc, score in reranked_results]
            
            logger.debug(f"RAG-Fusion search returned {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error in RAG-Fusion search: {e}")
            return []
    
    def hyde_search(self, query: str, k: int = 10) -> List[Document]:
        """Perform HyDE (Hypothetical Document Embeddings) search."""
        try:
            # Generate hypothetical document
            hypothetical_doc = self.query_transformer.generate_hypothetical_document(query)
            
            # Search using the hypothetical document
            results = self.vector_search(hypothetical_doc, k)
            
            logger.debug(f"HyDE search returned {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in HyDE search: {e}")
            return []
    
    def hybrid_search(self, query: str, route_config: Dict, 
                     page_context: Optional[Dict] = None) -> List[Document]:
        """Perform hybrid search based on route configuration."""
        try:
            all_results = []
            
            # Vector search
            vector_k = route_config.get('vector_search_k', 10)
            vector_results = self.vector_search(query, vector_k)
            all_results.append(vector_results)
            
            # Graph search if enabled
            if route_config.get('graph_search', False):
                graph_results = self.graph_search(query)
                if graph_results:
                    all_results.append(graph_results)
            
            # Query transformation if enabled
            if route_config.get('query_transformation', False):
                # Use RAG-Fusion for complex queries
                fusion_results = self.rag_fusion_search(query, vector_k)
                if fusion_results:
                    all_results.append(fusion_results)
            
            # Apply RRF to combine results
            if len(all_results) > 1:
                reranked_results = self.reciprocal_rank_fusion(all_results)
                documents = [doc for doc, score in reranked_results]
            else:
                documents = all_results[0] if all_results else []
            
            # Limit final results
            max_results = route_config.get('max_results', 10)
            documents = documents[:max_results]
            
            logger.info(f"Hybrid search returned {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def get_relevant_documents(self, query: str, route_config: Dict,
                             page_context: Optional[Dict] = None) -> List[Document]:
        """Main method to get relevant documents based on route configuration."""
        return self.hybrid_search(query, route_config, page_context)
    
    def search_with_fallback(self, query: str, route_config: Dict,
                           page_context: Optional[Dict] = None) -> List[Document]:
        """Search with fallback mechanisms if initial search fails."""
        try:
            # Try primary search method
            results = self.hybrid_search(query, route_config, page_context)
            
            # If results are insufficient, try fallback methods
            if len(results) < 3:
                logger.warning("Insufficient results, trying fallback methods")
                
                # Try broader vector search
                fallback_results = self.vector_search(query, k=20)
                if fallback_results:
                    results.extend(fallback_results)
                
                # Try graph search with broader scope
                if not route_config.get('graph_search', False):
                    graph_results = self.graph_search(query, max_depth=3)
                    if graph_results:
                        results.extend(graph_results)
                
                # Remove duplicates
                unique_results = []
                seen_content = set()
                
                for doc in results:
                    if doc.page_content not in seen_content:
                        unique_results.append(doc)
                        seen_content.add(doc.page_content)
                
                results = unique_results
            
            logger.info(f"Search with fallback returned {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in search with fallback: {e}")
            return []


def main():
    """Test the hybrid retriever."""
    # This would require initialized stores
    print("Hybrid retriever test - requires initialized vector and graph stores")
    print("Run this after setting up the stores in the main pipeline")


if __name__ == "__main__":
    main()
