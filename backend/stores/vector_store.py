"""
ChromaDB vector store implementation with multi-representation indexing.
"""
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
import structlog
from backend.config import settings

logger = structlog.get_logger(__name__)


class PyTorchVectorStore:
    """ChromaDB-based vector store with multi-representation indexing."""
    
    def __init__(self):
        self.persist_directory = Path(settings.chroma_persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize collections
        self.documents_collection = self._get_or_create_collection("pytorch_docs")
        self.summaries_collection = self._get_or_create_collection("pytorch_summaries")
        
        # Initialize multi-vector retriever components
        self.docstore = InMemoryByteStore()
        self.id_key = "doc_id"
        
        logger.info("Vector store initialized")
    
    def _get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection."""
        try:
            collection = self.client.get_collection(name)
            logger.info(f"Loaded existing collection: {name}")
        except ValueError:
            collection = self.client.create_collection(
                name=name,
                metadata={"description": f"PyTorch documentation {name}"}
            )
            logger.info(f"Created new collection: {name}")
        
        return collection
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def add_documents(self, documents: List[Document], collection_name: str = "documents") -> None:
        """Add documents to the specified collection."""
        if not documents:
            logger.warning("No documents provided")
            return
        
        collection = self.documents_collection if collection_name == "documents" else self.summaries_collection
        
        # Prepare data
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            texts.append(doc.page_content)
            
            # Prepare metadata for ChromaDB
            metadata = doc.metadata.copy()
            metadata[self.id_key] = doc_id
            metadatas.append(metadata)
        
        # Generate embeddings
        embeddings = self._embed_texts(texts)
        
        # Add to collection
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(documents)} documents to {collection_name} collection")
    
    def setup_multi_vector_retriever(self, documents: List[Document], 
                                   summaries: List[str]) -> MultiVectorRetriever:
        """Set up multi-vector retriever with documents and summaries."""
        # Create summary documents with doc_ids
        summary_docs = []
        doc_ids = []
        
        for i, summary in enumerate(summaries):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            summary_doc = Document(
                page_content=summary,
                metadata={self.id_key: doc_id}
            )
            summary_docs.append(summary_doc)
        
        # Add summaries to vector store
        self.add_documents(summary_docs, "summaries")
        
        # Store original documents in docstore
        self.docstore.mset(list(zip(doc_ids, documents)))
        
        # Create multi-vector retriever
        retriever = MultiVectorRetriever(
            vectorstore=self.summaries_collection,
            byte_store=self.docstore,
            id_key=self.id_key,
        )
        
        logger.info("Multi-vector retriever set up successfully")
        return retriever
    
    def similarity_search(self, query: str, k: int = 5, 
                         collection_name: str = "documents",
                         filter_dict: Optional[Dict] = None) -> List[Document]:
        """Perform similarity search in the specified collection."""
        collection = self.documents_collection if collection_name == "documents" else self.summaries_collection
        
        # Generate query embedding
        query_embedding = self._embed_texts([query])[0]
        
        # Build where clause for filtering
        where_clause = {}
        if filter_dict:
            where_clause = filter_dict
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause if where_clause else None
        )
        
        # Convert to Document objects
        documents = []
        if results['documents'] and results['documents'][0]:
            for i, content in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0.0
                
                # Add similarity score to metadata
                metadata['similarity_score'] = 1 - distance
                
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
        
        logger.debug(f"Found {len(documents)} similar documents for query")
        return documents
    
    def get_retriever(self, collection_name: str = "documents", 
                     search_kwargs: Optional[Dict] = None) -> chromadb.Collection:
        """Get a retriever for the specified collection."""
        collection = self.documents_collection if collection_name == "documents" else self.summaries_collection
        
        # Create a simple retriever wrapper
        class ChromaRetriever:
            def __init__(self, collection, search_kwargs=None):
                self.collection = collection
                self.search_kwargs = search_kwargs or {"k": 5}
                self.vector_store = None  # Reference to parent vector store
            
            def get_relevant_documents(self, query: str) -> List[Document]:
                k = self.search_kwargs.get("k", 5)
                return self.vector_store.similarity_search(query, k, collection.name)
            
            def similarity_search(self, query: str, k: int = 5) -> List[Document]:
                return self.vector_store.similarity_search(query, k, collection.name)
        
        retriever = ChromaRetriever(collection, search_kwargs)
        retriever.vector_store = self
        return retriever
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        docs_count = self.documents_collection.count()
        summaries_count = self.summaries_collection.count()
        
        return {
            "documents_count": docs_count,
            "summaries_count": summaries_count,
            "persist_directory": str(self.persist_directory),
            "embedding_model": "all-MiniLM-L6-v2"
        }
    
    def reset(self) -> None:
        """Reset the vector store (delete all data)."""
        try:
            self.client.delete_collection("pytorch_docs")
            self.client.delete_collection("pytorch_summaries")
            
            # Recreate collections
            self.documents_collection = self._get_or_create_collection("pytorch_docs")
            self.summaries_collection = self._get_or_create_collection("pytorch_summaries")
            
            logger.info("Vector store reset successfully")
        except Exception as e:
            logger.error(f"Error resetting vector store: {e}")
    
    def load_from_processed_data(self, processed_dir: Path) -> None:
        """Load documents from processed data files."""
        docs_file = processed_dir / "processed_documents.json"
        summaries_file = processed_dir / "document_summaries.json"
        
        if not docs_file.exists():
            logger.error(f"Processed documents file not found: {docs_file}")
            return
        
        # Load documents
        with open(docs_file, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
        
        documents = []
        for doc_data in docs_data:
            doc = Document(
                page_content=doc_data['page_content'],
                metadata=doc_data['metadata']
            )
            documents.append(doc)
        
        # Load summaries
        summaries = []
        if summaries_file.exists():
            with open(summaries_file, 'r', encoding='utf-8') as f:
                summaries = json.load(f)
        
        # Add documents to vector store
        self.add_documents(documents, "documents")
        
        # Set up multi-vector retriever if summaries available
        if summaries:
            self.setup_multi_vector_retriever(documents, summaries)
        
        logger.info(f"Loaded {len(documents)} documents and {len(summaries)} summaries")


def main():
    """Main function for testing the vector store."""
    from backend.config import settings
    
    # Initialize vector store
    vector_store = PyTorchVectorStore()
    
    # Test with sample data
    sample_docs = [
        Document(
            page_content="torch.tensor creates a tensor from data",
            metadata={"module": "torch", "function": "tensor", "url": "https://example.com"}
        ),
        Document(
            page_content="torch.nn.Linear applies a linear transformation",
            metadata={"module": "torch.nn", "class": "Linear", "url": "https://example.com"}
        )
    ]
    
    # Add documents
    vector_store.add_documents(sample_docs)
    
    # Test search
    results = vector_store.similarity_search("create tensor", k=2)
    print(f"Search results: {len(results)} documents")
    for doc in results:
        print(f"- {doc.metadata.get('function', 'unknown')}: {doc.page_content[:50]}...")
    
    # Print stats
    stats = vector_store.get_stats()
    print(f"Vector store stats: {stats}")


if __name__ == "__main__":
    main()
