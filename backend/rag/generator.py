"""
LLM generation with citation support and Self-RAG principles.
"""
import re
from typing import Dict, List, Optional, Tuple, Any
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import structlog
from backend.config import settings

logger = structlog.get_logger(__name__)


class CitationGenerator:
    """Generate citations for retrieved documents."""
    
    def __init__(self):
        logger.info("Citation generator initialized")
    
    def extract_citations(self, documents: List[Document]) -> List[Dict]:
        """Extract citation information from documents."""
        citations = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            
            citation = {
                "id": i,
                "title": metadata.get("title", "Untitled"),
                "url": metadata.get("source_url", metadata.get("url", "")),
                "module": metadata.get("module", ""),
                "function": metadata.get("function", ""),
                "class": metadata.get("class", ""),
                "relevance_score": metadata.get("relevance_score", 0.0)
            }
            
            citations.append(citation)
        
        return citations
    
    def format_citations(self, citations: List[Dict]) -> str:
        """Format citations for display."""
        if not citations:
            return ""
        
        citation_text = "\n\n**Sources:**\n"
        
        for citation in citations:
            citation_text += f"{citation['id']}. "
            
            if citation["title"]:
                citation_text += citation["title"]
            
            if citation["module"]:
                citation_text += f" (torch.{citation['module']})"
            
            if citation["url"]:
                citation_text += f" - [View Documentation]({citation['url']})"
            
            citation_text += "\n"
        
        return citation_text
    
    def add_citation_marks(self, text: str, documents: List[Document]) -> str:
        """Add citation marks to text based on document content."""
        if not documents:
            return text
        
        # Create citation mapping
        citation_map = {}
        for i, doc in enumerate(documents, 1):
            # Extract key phrases from document content
            key_phrases = self._extract_key_phrases(doc.page_content)
            for phrase in key_phrases:
                citation_map[phrase] = i
        
        # Add citation marks to text
        for phrase, citation_id in citation_map.items():
            if phrase in text and f"[{citation_id}]" not in text:
                text = text.replace(phrase, f"{phrase}[{citation_id}]")
        
        return text
    
    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases from document content."""
        # Simple phrase extraction (could be enhanced with NLP)
        phrases = []
        
        # Look for function names, class names, etc.
        patterns = [
            r'torch\.\w+\.\w+',  # torch.module.function
            r'nn\.\w+',          # nn.Linear
            r'optim\.\w+',       # optim.Adam
            r'F\.\w+',           # F.relu
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            phrases.extend(matches)
        
        return phrases[:5]  # Limit to 5 phrases per document


class SelfRAGGenerator:
    """Generator with Self-RAG principles for reflection and quality control."""
    
    def __init__(self):
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Generation prompt templates
        self.generation_template = ChatPromptTemplate.from_template("""
You are an expert PyTorch documentation assistant. Answer the user's question based ONLY on the provided context. Be accurate, helpful, and cite your sources.

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say so
3. Include code examples when relevant
4. Be concise but comprehensive
5. Use markdown formatting for code blocks

Answer:
""")
        
        self.reflection_template = ChatPromptTemplate.from_template("""
You are an expert evaluator. Assess the quality of the generated answer based on the context and question.

Question: {question}
Context: {context}
Answer: {answer}

Rate the answer on these criteria (1-5 scale):
1. Relevance: How well does the answer address the question?
2. Accuracy: Is the answer factually correct based on the context?
3. Completeness: Does the answer provide sufficient information?
4. Clarity: Is the answer clear and well-structured?

Provide your assessment:
""")
        
        self.citation_generator = CitationGenerator()
        
        logger.info("Self-RAG generator initialized")
    
    def _initialize_llm(self):
        """Initialize the local LLM."""
        if settings.use_local_model and settings.ollama_base_url:
            try:
                from langchain_ollama import Ollama
                llm = Ollama(
                    base_url=settings.ollama_base_url,
                    model=settings.ollama_model,
                    temperature=0.1  # Low temperature for consistent responses
                )
                logger.info(f"Initialized Ollama LLM: {settings.ollama_model}")
                return llm
            except ImportError:
                logger.warning("Ollama not available, falling back to mock LLM")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama: {e}")
        
        # Fallback to mock LLM
        logger.warning("Using mock LLM - install Ollama for production")
        return MockGeneratorLLM()
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents as context."""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            content = doc.page_content.strip()
            metadata = doc.metadata
            
            # Add source information
            source_info = []
            if metadata.get("module"):
                source_info.append(f"Module: torch.{metadata['module']}")
            if metadata.get("function"):
                source_info.append(f"Function: {metadata['function']}")
            if metadata.get("class"):
                source_info.append(f"Class: {metadata['class']}")
            
            context_part = f"**Source {i}**"
            if source_info:
                context_part += f" ({', '.join(source_info)})"
            context_part += f":\n{content}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate_answer(self, question: str, documents: List[Document]) -> Dict[str, Any]:
        """Generate answer with Self-RAG principles."""
        try:
            # Format context
            context = self.format_context(documents)
            
            # Generate initial answer
            generation_chain = self.generation_template | self.llm | StrOutputParser()
            answer = generation_chain.invoke({"question": question, "context": context})
            
            # Generate citations
            citations = self.citation_generator.extract_citations(documents)
            citation_text = self.citation_generator.format_citations(citations)
            
            # Add citation marks to answer
            answer_with_citations = self.citation_generator.add_citation_marks(answer, documents)
            
            # Reflect on answer quality
            reflection_result = self._reflect_on_answer(question, context, answer)
            
            # Compile final response
            response = {
                "answer": answer_with_citations,
                "citations": citations,
                "citation_text": citation_text,
                "reflection": reflection_result,
                "context_used": len(documents),
                "confidence": reflection_result.get("overall_score", 0.0)
            }
            
            logger.info(f"Generated answer with {len(documents)} context documents")
            return response
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": "I apologize, but I encountered an error while generating the answer.",
                "citations": [],
                "citation_text": "",
                "reflection": {"overall_score": 0.0, "error": str(e)},
                "context_used": 0,
                "confidence": 0.0
            }
    
    def _reflect_on_answer(self, question: str, context: str, answer: str) -> Dict:
        """Reflect on the quality of the generated answer."""
        try:
            reflection_chain = self.reflection_template | self.llm | StrOutputParser()
            reflection_text = reflection_chain.invoke({
                "question": question,
                "context": context,
                "answer": answer
            })
            
            # Parse reflection scores (simple extraction)
            scores = self._extract_reflection_scores(reflection_text)
            
            return {
                "reflection_text": reflection_text,
                "scores": scores,
                "overall_score": sum(scores.values()) / len(scores) if scores else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in reflection: {e}")
            return {
                "reflection_text": "Reflection failed",
                "scores": {},
                "overall_score": 0.5  # Default moderate score
            }
    
    def _extract_reflection_scores(self, reflection_text: str) -> Dict[str, float]:
        """Extract numerical scores from reflection text."""
        scores = {}
        
        # Look for patterns like "Relevance: 4" or "Relevance: 4/5"
        patterns = {
            "relevance": r"relevance[:\s]*(\d+)",
            "accuracy": r"accuracy[:\s]*(\d+)",
            "completeness": r"completeness[:\s]*(\d+)",
            "clarity": r"clarity[:\s]*(\d+)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, reflection_text.lower())
            if match:
                try:
                    score = int(match.group(1))
                    scores[key] = min(score / 5.0, 1.0)  # Normalize to 0-1
                except ValueError:
                    continue
        
        return scores
    
    def should_regenerate(self, reflection_result: Dict) -> bool:
        """Determine if the answer should be regenerated based on reflection."""
        overall_score = reflection_result.get("overall_score", 0.0)
        
        # Regenerate if overall score is below threshold
        return overall_score < 0.6
    
    def generate_with_retry(self, question: str, documents: List[Document],
                          max_retries: int = 2) -> Dict[str, Any]:
        """Generate answer with retry mechanism based on reflection."""
        best_response = None
        best_score = 0.0
        
        for attempt in range(max_retries + 1):
            try:
                response = self.generate_answer(question, documents)
                reflection = response.get("reflection", {})
                score = reflection.get("overall_score", 0.0)
                
                if score > best_score:
                    best_response = response
                    best_score = score
                
                # If score is good enough, return immediately
                if score >= 0.7:
                    logger.info(f"Generated satisfactory answer on attempt {attempt + 1}")
                    return response
                
                # If this is the last attempt, return best response
                if attempt == max_retries:
                    logger.warning(f"Reached max retries, returning best response (score: {best_score:.2f})")
                    return best_response
                
                logger.info(f"Answer quality below threshold (score: {score:.2f}), retrying...")
                
            except Exception as e:
                logger.error(f"Error in generation attempt {attempt + 1}: {e}")
                if attempt == max_retries:
                    return best_response or {
                        "answer": "I apologize, but I encountered an error while generating the answer.",
                        "citations": [],
                        "citation_text": "",
                        "reflection": {"overall_score": 0.0, "error": str(e)},
                        "context_used": 0,
                        "confidence": 0.0
                    }
        
        return best_response


class MockGeneratorLLM:
    """Mock LLM for testing when no real LLM is available."""
    
    def invoke(self, input_data: Dict) -> str:
        """Mock invoke method."""
        question = input_data.get("question", "")
        context = input_data.get("context", "")
        
        # Simple mock responses based on question content
        if "tensor" in question.lower():
            return """To create a tensor in PyTorch, you can use `torch.tensor()`:

```python
import torch

# Create tensor from list
data = [1, 2, 3, 4]
tensor = torch.tensor(data)
print(tensor)  # tensor([1, 2, 3, 4])

# Create tensor with specific dtype
tensor = torch.tensor(data, dtype=torch.float32)
```

The `torch.tensor()` function creates a tensor from various input types like lists, numpy arrays, or other tensors."""
        
        elif "neural" in question.lower() or "network" in question.lower():
            return """Here's how to create a neural network in PyTorch:

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model
model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
```

This creates a simple feedforward neural network with one hidden layer."""
        
        else:
            return f"""Based on the provided context, here's information about: {question}

The context contains relevant information that can help answer your question. Please refer to the source documents for more detailed information."""


def main():
    """Test the generator."""
    generator = SelfRAGGenerator()
    
    # Create test documents
    test_docs = [
        Document(
            page_content="torch.tensor creates a tensor from data. It can take lists, numpy arrays, or other tensors as input.",
            metadata={
                "function": "tensor",
                "module": "torch",
                "url": "https://pytorch.org/docs/stable/torch.html#torch.tensor"
            }
        ),
        Document(
            page_content="The dtype parameter specifies the data type of the tensor. Common dtypes include torch.float32, torch.int64, etc.",
            metadata={
                "parameter": "dtype",
                "function": "tensor",
                "module": "torch"
            }
        )
    ]
    
    question = "How do I create a tensor in PyTorch?"
    
    # Test generation
    response = generator.generate_answer(question, test_docs)
    
    print(f"Answer: {response['answer']}")
    print(f"Citations: {len(response['citations'])}")
    print(f"Confidence: {response['confidence']:.2f}")
    print(f"Reflection: {response['reflection']}")


if __name__ == "__main__":
    main()
