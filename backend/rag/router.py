"""
Intelligent routing for directing queries to appropriate processing paths.
"""
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import structlog
from backend.rag.query_transformer import QueryTransformer

logger = structlog.get_logger(__name__)


class RouteDecision(BaseModel):
    """Pydantic model for routing decisions."""
    
    route_type: Literal["simple_vector", "complex_hybrid", "graph_focused", "web_fallback"] = Field(
        ..., 
        description="The type of route to take for processing the query"
    )
    confidence: float = Field(
        ..., 
        description="Confidence in the routing decision (0.0 to 1.0)"
    )
    reasoning: str = Field(
        ..., 
        description="Brief explanation of the routing decision"
    )
    query_complexity: Literal["simple", "complex"] = Field(
        ..., 
        description="Assessed complexity of the query"
    )


class IntelligentRouter:
    """Intelligent router for query processing paths."""
    
    def __init__(self):
        self.query_transformer = QueryTransformer()
        
        # Routing prompt template
        self.routing_template = ChatPromptTemplate.from_template("""
You are an expert router for PyTorch documentation queries. Analyze the given query and determine the best processing route.

Available routes:
1. simple_vector: Direct vector search for straightforward queries
2. complex_hybrid: Full hybrid pipeline with vector + graph search
3. graph_focused: Focus on graph traversal for relationship-heavy queries
4. web_fallback: Web search for very specific or recent information

Query: {question}
Page Context: {page_context}
Query Complexity: {complexity}

Consider:
- Is this a simple lookup question? → simple_vector
- Does it involve relationships between PyTorch concepts? → graph_focused
- Is it a complex conceptual question? → complex_hybrid
- Is it about very recent features or specific error messages? → web_fallback

Route Decision:
""")
        
        # Route-specific processing configurations
        self.route_configs = {
            "simple_vector": {
                "vector_search_k": 5,
                "graph_search": False,
                "query_transformation": False,
                "reranking": False
            },
            "complex_hybrid": {
                "vector_search_k": 10,
                "graph_search": True,
                "query_transformation": True,
                "reranking": True
            },
            "graph_focused": {
                "vector_search_k": 3,
                "graph_search": True,
                "query_transformation": True,
                "reranking": False
            },
            "web_fallback": {
                "vector_search_k": 5,
                "graph_search": False,
                "query_transformation": False,
                "reranking": False,
                "web_search": True
            }
        }
        
        logger.info("Intelligent router initialized")
    
    def analyze_query_context(self, question: str, page_context: Dict) -> Dict:
        """Analyze the query in context of the current page."""
        context_info = {
            "has_pytorch_context": False,
            "current_module": "",
            "current_function": "",
            "current_class": "",
            "page_entities": [],
            "query_entities": []
        }
        
        # Extract context from page
        if page_context:
            page_content = page_context.get("content", "").lower()
            
            # Check for PyTorch context
            if any(keyword in page_content for keyword in ["torch", "pytorch", "nn", "tensor"]):
                context_info["has_pytorch_context"] = True
            
            # Extract current context
            context_info["current_module"] = page_context.get("module", "")
            context_info["current_function"] = page_context.get("function", "")
            context_info["current_class"] = page_context.get("class", "")
        
        # Extract entities from query
        question_lower = question.lower()
        pytorch_modules = ["torch", "nn", "optim", "utils", "vision", "audio"]
        context_info["query_entities"] = [
            module for module in pytorch_modules 
            if module in question_lower
        ]
        
        return context_info
    
    def determine_route(self, question: str, page_context: Optional[Dict] = None) -> RouteDecision:
        """Determine the best route for processing the query."""
        if page_context is None:
            page_context = {}
        
        # Analyze query complexity
        complexity = self.query_transformer.classify_query_complexity(question)
        
        # Analyze context
        context_info = self.analyze_query_context(question, page_context)
        
        # Determine route based on heuristics
        route_type = self._heuristic_routing(question, complexity, context_info)
        confidence = self._calculate_confidence(question, route_type, context_info)
        reasoning = self._generate_reasoning(question, route_type, context_info)
        
        return RouteDecision(
            route_type=route_type,
            confidence=confidence,
            reasoning=reasoning,
            query_complexity=complexity
        )
    
    def _heuristic_routing(self, question: str, complexity: str, context_info: Dict) -> str:
        """Use heuristics to determine routing."""
        question_lower = question.lower()
        
        # Check for specific patterns that indicate different routes
        
        # Web fallback indicators
        web_indicators = [
            "error", "bug", "issue", "problem", "not working", "broken",
            "latest", "new feature", "recent", "version", "update"
        ]
        if any(indicator in question_lower for indicator in web_indicators):
            return "web_fallback"
        
        # Graph-focused indicators
        graph_indicators = [
            "relationship", "related to", "similar to", "difference between",
            "compare", "vs", "versus", "inherit", "extends", "subclass"
        ]
        if any(indicator in question_lower for indicator in graph_indicators):
            return "graph_focused"
        
        # Simple vector indicators
        simple_indicators = [
            "what is", "definition", "meaning", "explain briefly",
            "syntax", "parameters", "return", "signature"
        ]
        if (complexity == "simple" and 
            any(indicator in question_lower for indicator in simple_indicators)):
            return "simple_vector"
        
        # Default to complex hybrid for complex queries
        if complexity == "complex":
            return "complex_hybrid"
        
        # Default fallback
        return "simple_vector"
    
    def _calculate_confidence(self, question: str, route_type: str, context_info: Dict) -> float:
        """Calculate confidence in the routing decision."""
        base_confidence = 0.7
        
        # Increase confidence based on context match
        if context_info["has_pytorch_context"]:
            base_confidence += 0.1
        
        # Increase confidence for specific patterns
        question_lower = question.lower()
        
        if route_type == "web_fallback":
            web_indicators = ["error", "bug", "latest", "new"]
            if any(indicator in question_lower for indicator in web_indicators):
                base_confidence += 0.2
        
        elif route_type == "graph_focused":
            graph_indicators = ["relationship", "compare", "difference"]
            if any(indicator in question_lower for indicator in graph_indicators):
                base_confidence += 0.2
        
        elif route_type == "simple_vector":
            simple_indicators = ["what is", "definition", "syntax"]
            if any(indicator in question_lower for indicator in simple_indicators):
                base_confidence += 0.2
        
        # Cap at 1.0
        return min(base_confidence, 1.0)
    
    def _generate_reasoning(self, question: str, route_type: str, context_info: Dict) -> str:
        """Generate reasoning for the routing decision."""
        reasoning_parts = []
        
        if route_type == "simple_vector":
            reasoning_parts.append("Simple lookup query detected")
        
        elif route_type == "complex_hybrid":
            reasoning_parts.append("Complex query requiring comprehensive search")
        
        elif route_type == "graph_focused":
            reasoning_parts.append("Query involves relationships between concepts")
        
        elif route_type == "web_fallback":
            reasoning_parts.append("Query may require recent or specific information")
        
        if context_info["has_pytorch_context"]:
            reasoning_parts.append("Page context available for enhanced search")
        
        if context_info["query_entities"]:
            reasoning_parts.append(f"Detected entities: {', '.join(context_info['query_entities'])}")
        
        return "; ".join(reasoning_parts)
    
    def get_route_config(self, route_type: str) -> Dict:
        """Get configuration for the specified route."""
        return self.route_configs.get(route_type, self.route_configs["simple_vector"])
    
    def should_use_query_transformation(self, route_type: str) -> bool:
        """Determine if query transformation should be used."""
        config = self.get_route_config(route_type)
        return config.get("query_transformation", False)
    
    def should_use_graph_search(self, route_type: str) -> bool:
        """Determine if graph search should be used."""
        config = self.get_route_config(route_type)
        return config.get("graph_search", False)
    
    def should_use_reranking(self, route_type: str) -> bool:
        """Determine if reranking should be used."""
        config = self.get_route_config(route_type)
        return config.get("reranking", False)
    
    def get_vector_search_k(self, route_type: str) -> int:
        """Get the number of documents to retrieve for vector search."""
        config = self.get_route_config(route_type)
        return config.get("vector_search_k", 5)


def main():
    """Test the intelligent router."""
    router = IntelligentRouter()
    
    test_cases = [
        {
            "question": "What is torch.tensor?",
            "page_context": {"module": "torch", "content": "torch tensor operations"}
        },
        {
            "question": "How to implement a neural network with PyTorch?",
            "page_context": {}
        },
        {
            "question": "What's the difference between nn.Linear and nn.Conv2d?",
            "page_context": {"module": "torch.nn"}
        },
        {
            "question": "torch.tensor error: expected scalar type Float but found Long",
            "page_context": {}
        }
    ]
    
    for test_case in test_cases:
        decision = router.determine_route(
            test_case["question"], 
            test_case["page_context"]
        )
        
        print(f"\nQuestion: {test_case['question']}")
        print(f"Route: {decision.route_type}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Reasoning: {decision.reasoning}")
        
        config = router.get_route_config(decision.route_type)
        print(f"Config: {config}")


if __name__ == "__main__":
    main()
