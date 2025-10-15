"""
Document processor for chunking PyTorch documentation and extracting metadata.
"""
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import structlog
from backend.config import settings

logger = structlog.get_logger(__name__)


class DocumentProcessor:
    """Processes PyTorch documentation into chunks with metadata preservation."""
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize tokenizer for more precise chunking
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken: {e}")
            self.tokenizer = None
    
    def load_scraped_document(self, file_path: Path) -> Optional[Dict]:
        """Load a scraped document from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            return None
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract PyTorch entities (modules, classes, functions) from text."""
        entities = {
            'modules': [],
            'classes': [],
            'functions': [],
            'parameters': []
        }
        
        # Simple regex-based extraction (could be enhanced with NER)
        import re
        
        # Extract module imports
        module_pattern = r'torch\.(\w+(?:\.\w+)*)'
        entities['modules'] = list(set(re.findall(module_pattern, text)))
        
        # Extract class definitions (looking for patterns like "class ClassName")
        class_pattern = r'class\s+([A-Z][a-zA-Z0-9_]*)'
        entities['classes'] = list(set(re.findall(class_pattern, text)))
        
        # Extract function definitions
        func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        entities['functions'] = list(set(re.findall(func_pattern, text)))
        
        # Extract parameter names in function signatures
        param_pattern = r'(\w+)\s*[:=]'
        entities['parameters'] = list(set(re.findall(param_pattern, text)))
        
        return entities
    
    def create_chunk_metadata(self, chunk: str, doc_metadata: Dict, chunk_index: int, 
                            entities: Dict, code_examples: List[Dict]) -> Dict:
        """Create comprehensive metadata for a document chunk."""
        metadata = {
            'chunk_id': str(uuid.uuid4()),
            'chunk_index': chunk_index,
            'source_url': doc_metadata.get('url', ''),
            'title': doc_metadata.get('title', ''),
            'module': doc_metadata.get('module', ''),
            'function': doc_metadata.get('function', ''),
            'class': doc_metadata.get('class', ''),
            'description': doc_metadata.get('description', ''),
            'scraped_at': doc_metadata.get('scraped_at', 0),
            'chunk_length': len(chunk),
            'entities': entities,
            'has_code': any(example['code'] in chunk for example in code_examples),
            'code_examples_count': len([ex for ex in code_examples if ex['code'] in chunk])
        }
        
        # Add token count if tokenizer available
        if self.tokenizer:
            metadata['token_count'] = len(self.tokenizer.encode(chunk))
        
        return metadata
    
    def process_document(self, doc_data: Dict) -> List[Document]:
        """Process a single document into chunks."""
        if not doc_data:
            return []
        
        metadata = doc_data.get('metadata', {})
        content = doc_data.get('content', '')
        code_examples = doc_data.get('code_examples', [])
        
        if not content:
            logger.warning(f"No content found in document: {metadata.get('url', 'unknown')}")
            return []
        
        # Extract entities from the full document
        entities = self.extract_entities(content)
        
        # Split content into chunks
        chunks = self.text_splitter.split_text(content)
        
        # Convert to LangChain Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue
            
            chunk_metadata = self.create_chunk_metadata(
                chunk, metadata, i, entities, code_examples
            )
            
            doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        logger.info(f"Processed document into {len(documents)} chunks")
        return documents
    
    def create_summary(self, doc_data: Dict) -> str:
        """Create a summary of the document for multi-representation indexing."""
        metadata = doc_data.get('metadata', {})
        content = doc_data.get('content', '')
        
        # Extract key information
        title = metadata.get('title', '')
        module = metadata.get('module', '')
        function = metadata.get('function', '')
        description = metadata.get('description', '')
        
        # Create structured summary
        summary_parts = []
        
        if title:
            summary_parts.append(f"Title: {title}")
        
        if module:
            summary_parts.append(f"Module: torch.{module}")
        
        if function:
            summary_parts.append(f"Function: {function}")
        
        if description:
            summary_parts.append(f"Description: {description}")
        
        # Add first few sentences of content
        sentences = content.split('. ')[:3]
        if sentences:
            summary_parts.append(f"Content: {' '.join(sentences)}...")
        
        return '\n'.join(summary_parts)
    
    def process_directory(self, docs_dir: Path) -> Tuple[List[Document], List[str]]:
        """Process all documents in a directory."""
        all_documents = []
        all_summaries = []
        
        if not docs_dir.exists():
            logger.error(f"Directory does not exist: {docs_dir}")
            return all_documents, all_summaries
        
        # Find all JSON files
        json_files = list(docs_dir.rglob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        for file_path in json_files:
            try:
                doc_data = self.load_scraped_document(file_path)
                if doc_data:
                    # Process document into chunks
                    documents = self.process_document(doc_data)
                    all_documents.extend(documents)
                    
                    # Create summary
                    summary = self.create_summary(doc_data)
                    all_summaries.append(summary)
                    
                    logger.debug(f"Processed {file_path.name}: {len(documents)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Total processed: {len(all_documents)} chunks, {len(all_summaries)} summaries")
        return all_documents, all_summaries
    
    def save_processed_data(self, documents: List[Document], summaries: List[str], 
                          output_dir: Path) -> None:
        """Save processed documents and summaries to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save documents
        docs_file = output_dir / "processed_documents.json"
        docs_data = []
        
        for doc in documents:
            docs_data.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })
        
        with open(docs_file, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, indent=2, ensure_ascii=False)
        
        # Save summaries
        summaries_file = output_dir / "document_summaries.json"
        with open(summaries_file, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(documents)} documents and {len(summaries)} summaries")


def main():
    """Main function for processing documents."""
    from backend.config import settings
    
    docs_dir = Path(settings.pytorch_docs_dir)
    output_dir = Path(settings.cache_dir) / "processed"
    
    processor = DocumentProcessor()
    documents, summaries = processor.process_directory(docs_dir)
    
    if documents:
        processor.save_processed_data(documents, summaries, output_dir)
        print(f"Processed {len(documents)} document chunks and {len(summaries)} summaries")
    else:
        print("No documents found to process")


if __name__ == "__main__":
    main()
