"""
Advanced query transformation techniques for improved retrieval.
"""
import asyncio
from typing import Dict, List, Optional, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import structlog
from backend.config import settings

logger = structlog.get_logger(__name__)


class QueryTransformer:
    """Advanced query transformation using local LLMs."""
    
    def __init__(self):
        # Initialize local LLM (using Ollama as default)
        self.llm = self._initialize_llm()
        
        # Query transformation templates
        self.multi_query_template = ChatPromptTemplate.from_template("""
You are an AI assistant specialized in PyTorch documentation. Your task is to generate 4 different versions of the given user question to retrieve relevant documents from a vector database.

By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.

Provide these alternative questions separated by newlines. Focus on:
1. Different terminology and synonyms
2. More specific technical terms
3. Broader conceptual approaches
4. Different use cases or contexts

Original question: {question}

Alternative questions:
""")
        
        self.rag_fusion_template = ChatPromptTemplate.from_template("""
You are a helpful assistant that generates multiple search queries based on a single input query for PyTorch documentation.

Generate multiple search queries related to: {question}

Focus on different aspects:
- Technical implementation details
- Conceptual understanding
- Use cases and examples
- Related PyTorch modules or functions

Output (4 queries):
""")
        
        self.decomposition_template = ChatPromptTemplate.from_template("""
You are a helpful assistant that generates multiple sub-questions related to an input question about PyTorch.

The goal is to break down the input into a set of sub-problems/sub-questions that can be answered in isolation.

Generate multiple search queries related to: {question}

Output (3 queries):
""")
        
        self.step_back_template = ChatPromptTemplate.from_template("""
You are an expert at PyTorch documentation. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer.

Here are a few examples:
- Input: "Could the members of The Police perform lawful arrests?"
- Output: "what can the members of The Police do?"

- Input: "Jan Sindel's was born in what country?"
- Output: "what is Jan Sindel's personal history?"

Original question: {question}

Step-back question:
""")
        
        self.hyde_template = ChatPromptTemplate.from_template("""
Please write a scientific paper passage to answer the question about PyTorch.

Question: {question}

Passage:
""")
        
        logger.info("Query transformer initialized")
    
    def _initialize_llm(self):
        """Initialize the local LLM."""
        if settings.use_local_model and settings.ollama_base_url:
            try:
                # Try Ollama first
                from langchain_ollama import Ollama
                llm = Ollama(
                    base_url=settings.ollama_base_url,
                    model=settings.ollama_model,
                    temperature=0
                )
                logger.info(f"Initialized Ollama LLM: {settings.ollama_model}")
                return llm
            except ImportError:
                logger.warning("Ollama not available, falling back to mock LLM")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama: {e}")
        
        # Fallback to a mock LLM for testing
        logger.warning("Using mock LLM - install Ollama or OpenAI for production")
        return MockLLM()
    
    def generate_multi_queries(self, question: str) -> List[str]:
        """Generate multiple alternative queries using Multi-Query approach."""
        try:
            chain = self.multi_query_template | self.llm | StrOutputParser()
            response = chain.invoke({"question": question})
            
            # Split response into individual queries
            queries = [q.strip() for q in response.split('\n') if q.strip()]
            
            # Ensure we have at least the original question
            if not queries:
                queries = [question]
            
            # Limit to 4 queries
            queries = queries[:4]
            
            logger.debug(f"Generated {len(queries)} multi-queries")
            return queries
            
        except Exception as e:
            logger.error(f"Error generating multi-queries: {e}")
            return [question]
    
    def generate_rag_fusion_queries(self, question: str) -> List[str]:
        """Generate queries optimized for RAG-Fusion."""
        try:
            chain = self.rag_fusion_template | self.llm | StrOutputParser()
            response = chain.invoke({"question": question})
            
            queries = [q.strip() for q in response.split('\n') if q.strip()]
            
            if not queries:
                queries = [question]
            
            queries = queries[:4]
            
            logger.debug(f"Generated {len(queries)} RAG-Fusion queries")
            return queries
            
        except Exception as e:
            logger.error(f"Error generating RAG-Fusion queries: {e}")
            return [question]
    
    def decompose_query(self, question: str) -> List[str]:
        """Decompose complex queries into simpler sub-questions."""
        try:
            chain = self.decomposition_template | self.llm | StrOutputParser()
            response = chain.invoke({"question": question})
            
            sub_questions = [q.strip() for q in response.split('\n') if q.strip()]
            
            if not sub_questions:
                sub_questions = [question]
            
            sub_questions = sub_questions[:3]
            
            logger.debug(f"Decomposed into {len(sub_questions)} sub-questions")
            return sub_questions
            
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            return [question]
    
    def generate_step_back_query(self, question: str) -> str:
        """Generate a step-back (more general) version of the query."""
        try:
            chain = self.step_back_template | self.llm | StrOutputParser()
            step_back_question = chain.invoke({"question": question})
            
            logger.debug(f"Generated step-back query: {step_back_question}")
            return step_back_question.strip()
            
        except Exception as e:
            logger.error(f"Error generating step-back query: {e}")
            return question
    
    def generate_hypothetical_document(self, question: str) -> str:
        """Generate a hypothetical document for HyDE approach."""
        try:
            chain = self.hyde_template | self.llm | StrOutputParser()
            hypothetical_doc = chain.invoke({"question": question})
            
            logger.debug(f"Generated hypothetical document ({len(hypothetical_doc)} chars)")
            return hypothetical_doc
            
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {e}")
            return question
    
    def classify_query_complexity(self, question: str) -> str:
        """Classify query as simple or complex."""
        complexity_indicators = [
            "how", "why", "what is", "explain", "difference", "compare",
            "implement", "create", "build", "optimize", "debug", "troubleshoot"
        ]
        
        question_lower = question.lower()
        
        # Check for complexity indicators
        has_indicators = any(indicator in question_lower for indicator in complexity_indicators)
        
        # Check length and structure
        is_long = len(question.split()) > 8
        has_multiple_parts = " and " in question_lower or " or " in question_lower
        
        if has_indicators or is_long or has_multiple_parts:
            return "complex"
        else:
            return "simple"
    
    def get_query_strategy(self, question: str) -> Dict[str, any]:
        """Determine the best query transformation strategy."""
        complexity = self.classify_query_complexity(question)
        
        if complexity == "simple":
            return {
                "strategy": "direct",
                "queries": [question],
                "transformation": "none"
            }
        else:
            # For complex queries, use multiple strategies
            return {
                "strategy": "multi_transform",
                "queries": self.generate_multi_queries(question),
                "transformation": "multi_query",
                "step_back": self.generate_step_back_query(question),
                "hyde": self.generate_hypothetical_document(question)
            }


class MockLLM:
    """Mock LLM for testing when no real LLM is available."""
    
    def invoke(self, input_data: Dict) -> str:
        """Mock invoke method."""
        question = input_data.get("question", "")
        
        # Simple mock responses based on keywords
        if "tensor" in question.lower():
            return "torch.Tensor\npytorch tensor operations\ntensor creation\nmultidimensional arrays"
        elif "neural" in question.lower() or "network" in question.lower():
            return "neural networks\npytorch nn module\ndeep learning\ntraining models"
        elif "optimization" in question.lower():
            return "optimizers\ngradient descent\nloss functions\nmodel training"
        else:
            return f"Alternative query 1: {question}\nAlternative query 2: pytorch {question}\nAlternative query 3: how to {question}\nAlternative query 4: {question} examples"


def main():
    """Test the query transformer."""
    transformer = QueryTransformer()
    
    test_questions = [
        "How do I create a tensor?",
        "What is the difference between torch.nn.Linear and torch.nn.Conv2d?",
        "How to implement a neural network with PyTorch?",
        "tensor operations"
    ]
    
    for question in test_questions:
        print(f"\nOriginal: {question}")
        
        # Test different transformations
        multi_queries = transformer.generate_multi_queries(question)
        print(f"Multi-queries: {multi_queries}")
        
        strategy = transformer.get_query_strategy(question)
        print(f"Strategy: {strategy['strategy']}")
        
        if strategy['strategy'] == 'multi_transform':
            print(f"Step-back: {strategy['step_back']}")


if __name__ == "__main__":
    main()
