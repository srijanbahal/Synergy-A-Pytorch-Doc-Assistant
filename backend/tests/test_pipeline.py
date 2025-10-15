"""
End-to-end testing for the PyTorch RAG Assistant pipeline.
"""
import pytest
import asyncio
import time
from typing import Dict, Any
import structlog
from backend.rag.pipeline import RAGPipeline
from backend.stores.vector_store import ChromaVectorStore
from backend.stores.graph_store import Neo4jGraphStore
from backend.context.page_analyzer import PageAnalyzer
from backend.memory.chat_history import ConversationMemory
from backend.config import settings

logger = structlog.get_logger(__name__)


class TestPipeline:
    """Test suite for the RAG pipeline."""
    
    def __init__(self):
        self.vector_store = None
        self.graph_store = None
        self.pipeline = None
        self.page_analyzer = None
        self.conversation_memory = None
        self.test_results = {}
    
    async def setup(self):
        """Setup test environment."""
        try:
            # Initialize stores
            self.vector_store = ChromaVectorStore()
            self.graph_store = Neo4jGraphStore()
            
            # Initialize pipeline
            self.pipeline = RAGPipeline(self.vector_store, self.graph_store)
            
            # Initialize other components
            self.page_analyzer = PageAnalyzer()
            self.conversation_memory = ConversationMemory()
            
            logger.info("Test setup completed successfully")
            
        except Exception as e:
            logger.error(f"Test setup failed: {e}")
            raise
    
    async def test_basic_query(self):
        """Test basic query processing."""
        test_queries = [
            "How do I create a tensor in PyTorch?",
            "What is torch.nn.Linear?",
            "How to use torch.optim.Adam?",
            "What is the difference between torch.tensor and torch.Tensor?",
            "How to implement a neural network?"
        ]
        
        results = []
        for query in test_queries:
            try:
                start_time = time.time()
                result = self.pipeline.process_query(query)
                processing_time = time.time() - start_time
                
                results.append({
                    'query': query,
                    'success': True,
                    'processing_time': processing_time,
                    'confidence': result.get('confidence', 0.0),
                    'answer_length': len(result.get('answer', '')),
                    'citations_count': len(result.get('citations', []))
                })
                
                logger.info(f"Query processed: {query[:50]}... (Time: {processing_time:.2f}s, Confidence: {result.get('confidence', 0.0):.2f})")
                
            except Exception as e:
                results.append({
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'processing_time': 0.0
                })
                logger.error(f"Query failed: {query[:50]}... Error: {e}")
        
        self.test_results['basic_queries'] = results
        return results
    
    async def test_page_context_processing(self):
        """Test page context analysis."""
        sample_html = """
        <html>
            <head><title>torch.nn.Linear - PyTorch Documentation</title></head>
            <body>
                <h1>torch.nn.Linear</h1>
                <p>Applies a linear transformation to the incoming data: y = xA^T + b</p>
                <h2>Parameters</h2>
                <ul>
                    <li>in_features: size of each input sample</li>
                    <li>out_features: size of each output sample</li>
                </ul>
                <pre><code>import torch
linear = torch.nn.Linear(in_features=784, out_features=128)
output = linear(input_tensor)</code></pre>
            </body>
        </html>
        """
        
        try:
            context = self.page_analyzer.analyze_page(sample_html, "https://pytorch.org/docs/stable/generated/torch.nn.Linear.html")
            
            result = {
                'success': True,
                'context': context,
                'entities_found': len(context.get('entities', {})),
                'code_examples': len(context.get('code_examples', [])),
                'current_function': context.get('current_function'),
                'current_class': context.get('current_class')
            }
            
            logger.info(f"Page context analyzed: {result['entities_found']} entities, {result['code_examples']} code examples")
            
        except Exception as e:
            result = {
                'success': False,
                'error': str(e)
            }
            logger.error(f"Page context analysis failed: {e}")
        
        self.test_results['page_context'] = result
        return result
    
    async def test_conversation_memory(self):
        """Test conversation memory functionality."""
        session_id = "test_session_123"
        
        try:
            # Add messages
            self.conversation_memory.add_message(session_id, {
                "type": "user",
                "content": "How do I create a tensor?"
            })
            
            self.conversation_memory.add_message(session_id, {
                "type": "assistant",
                "content": "You can create a tensor using torch.tensor(data)."
            })
            
            # Get conversation history
            history = self.conversation_memory.get_conversation_history(session_id)
            
            # Get context
            context = self.conversation_memory.get_conversation_context(session_id)
            
            # Get session info
            session_info = self.conversation_memory.get_session_info(session_id)
            
            result = {
                'success': True,
                'history_length': len(history),
                'context_length': len(context),
                'session_info': session_info
            }
            
            logger.info(f"Conversation memory test: {result['history_length']} messages, context length: {result['context_length']}")
            
        except Exception as e:
            result = {
                'success': False,
                'error': str(e)
            }
            logger.error(f"Conversation memory test failed: {e}")
        
        self.test_results['conversation_memory'] = result
        return result
    
    async def test_performance_metrics(self):
        """Test performance metrics and optimization."""
        test_queries = [
            "What is PyTorch?",
            "How to use torch.nn.Linear?",
            "What are the parameters of torch.optim.Adam?",
            "How to implement backpropagation?",
            "What is torch.autograd?"
        ]
        
        performance_metrics = []
        
        for query in test_queries:
            try:
                start_time = time.time()
                result = self.pipeline.process_query(query)
                total_time = time.time() - start_time
                
                pipeline_metrics = result.get('pipeline_metrics', {})
                
                performance_metrics.append({
                    'query': query,
                    'total_time': total_time,
                    'routing_time': pipeline_metrics.get('routing_time', 0.0),
                    'retrieval_time': pipeline_metrics.get('retrieval_time', 0.0),
                    'generation_time': pipeline_metrics.get('generation_time', 0.0),
                    'documents_retrieved': pipeline_metrics.get('documents_retrieved', 0),
                    'confidence': result.get('confidence', 0.0)
                })
                
            except Exception as e:
                performance_metrics.append({
                    'query': query,
                    'error': str(e),
                    'total_time': 0.0
                })
        
        # Calculate averages
        successful_queries = [m for m in performance_metrics if 'error' not in m]
        
        if successful_queries:
            avg_metrics = {
                'average_total_time': sum(m['total_time'] for m in successful_queries) / len(successful_queries),
                'average_routing_time': sum(m['routing_time'] for m in successful_queries) / len(successful_queries),
                'average_retrieval_time': sum(m['retrieval_time'] for m in successful_queries) / len(successful_queries),
                'average_generation_time': sum(m['generation_time'] for m in successful_queries) / len(successful_queries),
                'average_confidence': sum(m['confidence'] for m in successful_queries) / len(successful_queries),
                'success_rate': len(successful_queries) / len(performance_metrics)
            }
        else:
            avg_metrics = {'error': 'No successful queries'}
        
        result = {
            'performance_metrics': performance_metrics,
            'average_metrics': avg_metrics
        }
        
        self.test_results['performance'] = result
        return result
    
    async def test_error_handling(self):
        """Test error handling and edge cases."""
        error_test_cases = [
            "",  # Empty query
            "a" * 1000,  # Very long query
            "What is torch.nonexistent.function?",  # Non-existent function
            "How to implement quantum computing?",  # Unrelated query
            "torch.tensor([1, 2, 3])" * 100,  # Code-only query
        ]
        
        error_results = []
        
        for query in error_test_cases:
            try:
                result = self.pipeline.process_query(query)
                error_results.append({
                    'query': query[:100] + "..." if len(query) > 100 else query,
                    'handled': True,
                    'confidence': result.get('confidence', 0.0),
                    'answer_length': len(result.get('answer', ''))
                })
                
            except Exception as e:
                error_results.append({
                    'query': query[:100] + "..." if len(query) > 100 else query,
                    'handled': False,
                    'error': str(e)
                })
        
        result = {
            'error_test_cases': error_results,
            'handled_cases': len([r for r in error_results if r['handled']]),
            'total_cases': len(error_results)
        }
        
        self.test_results['error_handling'] = result
        return result
    
    async def run_all_tests(self):
        """Run all tests and generate report."""
        logger.info("Starting comprehensive pipeline testing")
        
        await self.setup()
        
        # Run all tests
        await self.test_basic_query()
        await self.test_page_context_processing()
        await self.test_conversation_memory()
        await self.test_performance_metrics()
        await self.test_error_handling()
        
        # Generate test report
        report = self.generate_test_report()
        
        logger.info("All tests completed")
        return report
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("# PyTorch RAG Assistant - Test Report")
        report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic queries test
        basic_queries = self.test_results.get('basic_queries', [])
        if basic_queries:
            successful = len([q for q in basic_queries if q['success']])
            avg_time = sum(q['processing_time'] for q in basic_queries if q['success']) / max(successful, 1)
            avg_confidence = sum(q['confidence'] for q in basic_queries if q['success']) / max(successful, 1)
            
            report.append("## Basic Query Processing")
            report.append(f"- **Success Rate**: {successful}/{len(basic_queries)} ({successful/len(basic_queries)*100:.1f}%)")
            report.append(f"- **Average Processing Time**: {avg_time:.2f}s")
            report.append(f"- **Average Confidence**: {avg_confidence:.3f}")
            report.append("")
        
        # Page context test
        page_context = self.test_results.get('page_context', {})
        if page_context:
            report.append("## Page Context Analysis")
            if page_context['success']:
                report.append(f"- **Entities Found**: {page_context['entities_found']}")
                report.append(f"- **Code Examples**: {page_context['code_examples']}")
                report.append(f"- **Current Function**: {page_context.get('current_function', 'None')}")
                report.append(f"- **Current Class**: {page_context.get('current_class', 'None')}")
            else:
                report.append(f"- **Status**: Failed - {page_context['error']}")
            report.append("")
        
        # Conversation memory test
        conversation_memory = self.test_results.get('conversation_memory', {})
        if conversation_memory:
            report.append("## Conversation Memory")
            if conversation_memory['success']:
                report.append(f"- **History Length**: {conversation_memory['history_length']} messages")
                report.append(f"- **Context Length**: {conversation_memory['context_length']} characters")
                report.append(f"- **Session Info**: Available")
            else:
                report.append(f"- **Status**: Failed - {conversation_memory['error']}")
            report.append("")
        
        # Performance metrics
        performance = self.test_results.get('performance', {})
        if performance and 'average_metrics' in performance:
            avg_metrics = performance['average_metrics']
            if 'error' not in avg_metrics:
                report.append("## Performance Metrics")
                report.append(f"- **Average Total Time**: {avg_metrics['average_total_time']:.2f}s")
                report.append(f"- **Average Routing Time**: {avg_metrics['average_routing_time']:.2f}s")
                report.append(f"- **Average Retrieval Time**: {avg_metrics['average_retrieval_time']:.2f}s")
                report.append(f"- **Average Generation Time**: {avg_metrics['average_generation_time']:.2f}s")
                report.append(f"- **Average Confidence**: {avg_metrics['average_confidence']:.3f}")
                report.append(f"- **Success Rate**: {avg_metrics['success_rate']*100:.1f}%")
            report.append("")
        
        # Error handling
        error_handling = self.test_results.get('error_handling', {})
        if error_handling:
            report.append("## Error Handling")
            report.append(f"- **Handled Cases**: {error_handling['handled_cases']}/{error_handling['total_cases']}")
            report.append(f"- **Error Handling Rate**: {error_handling['handled_cases']/error_handling['total_cases']*100:.1f}%")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        
        if basic_queries:
            successful = len([q for q in basic_queries if q['success']])
            if successful / len(basic_queries) < 0.8:
                report.append("- Improve query processing success rate")
        
        if performance and 'average_metrics' in performance:
            avg_metrics = performance['average_metrics']
            if 'error' not in avg_metrics:
                if avg_metrics['average_total_time'] > 5.0:
                    report.append("- Optimize pipeline performance")
                if avg_metrics['average_confidence'] < 0.7:
                    report.append("- Improve answer quality and confidence")
        
        if error_handling:
            if error_handling['handled_cases'] / error_handling['total_cases'] < 0.8:
                report.append("- Improve error handling for edge cases")
        
        return "\n".join(report)


async def main():
    """Main function for running tests."""
    tester = TestPipeline()
    report = await tester.run_all_tests()
    
    print(report)
    
    # Save report
    report_file = f"test_report_{int(time.time())}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nTest report saved to: {report_file}")


if __name__ == "__main__":
    asyncio.run(main())
