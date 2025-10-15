"""
RAGAS evaluation pipeline for comprehensive RAG system assessment.
"""
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_correctness,
    context_precision
)
import structlog
from backend.rag.pipeline import get_pipeline
from backend.config import settings

logger = structlog.get_logger(__name__)


class RAGASEvaluator:
    """RAGAS evaluation system for PyTorch RAG Assistant."""
    
    def __init__(self):
        self.pipeline = get_pipeline()
        self.dataset_path = Path(settings.ragas_dataset_path)
        self.cache_dir = Path(settings.evaluation_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # RAGAS metrics
        self.metrics = [
            faithfulness,           # How factually consistent is the answer with the context?
            answer_relevancy,       # How relevant is the answer to the question?
            context_recall,         # Did we retrieve all the necessary context?
            answer_correctness,     # How accurate is the answer compared to ground truth?
            context_precision,      # How precise is the retrieved context?
        ]
        
        logger.info("RAGAS evaluator initialized")
    
    def load_test_dataset(self) -> Dataset:
        """Load the test dataset for evaluation."""
        if not self.dataset_path.exists():
            logger.warning(f"Test dataset not found at {self.dataset_path}")
            return self.create_sample_dataset()
        
        try:
            with open(self.dataset_path, 'r') as f:
                data = json.load(f)
            
            # Convert to HuggingFace Dataset format
            dataset = Dataset.from_dict(data)
            logger.info(f"Loaded test dataset with {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading test dataset: {e}")
            return self.create_sample_dataset()
    
    def create_sample_dataset(self) -> Dataset:
        """Create a sample test dataset for evaluation."""
        sample_data = {
            'question': [
                "How do I create a tensor in PyTorch?",
                "What is the difference between torch.nn.Linear and torch.nn.Conv2d?",
                "How to implement a neural network with PyTorch?",
                "What are the parameters of torch.optim.Adam?",
                "How to use torch.nn.functional.relu?",
                "What is backpropagation in PyTorch?",
                "How to save and load a PyTorch model?",
                "What is the difference between torch.tensor and torch.Tensor?",
                "How to use torch.nn.Dropout?",
                "What are the common activation functions in PyTorch?"
            ],
            'answer': [
                "You can create a tensor using torch.tensor(data). For example: torch.tensor([1, 2, 3]) creates a tensor from a list.",
                "torch.nn.Linear applies a linear transformation, while torch.nn.Conv2d applies a 2D convolution. Linear is for fully connected layers, Conv2d for convolutional layers.",
                "Create a class inheriting from nn.Module, define layers in __init__, implement forward method, then instantiate and use for training/inference.",
                "torch.optim.Adam parameters include: params (model parameters), lr (learning rate), betas (momentum coefficients), eps (epsilon), weight_decay, amsgrad.",
                "torch.nn.functional.relu applies the ReLU activation function element-wise. Usage: torch.nn.functional.relu(x) or F.relu(x).",
                "Backpropagation in PyTorch is automatic through autograd. Call loss.backward() to compute gradients, then optimizer.step() to update parameters.",
                "Save: torch.save(model.state_dict(), 'model.pth'). Load: model.load_state_dict(torch.load('model.pth')).",
                "torch.tensor is a function that creates tensors, torch.Tensor is a class. torch.tensor is preferred as it's more explicit about data types.",
                "torch.nn.Dropout randomly sets input elements to zero during training. Usage: torch.nn.Dropout(p=0.5) where p is dropout probability.",
                "Common activation functions: ReLU (torch.nn.ReLU), Sigmoid (torch.nn.Sigmoid), Tanh (torch.nn.Tanh), LeakyReLU (torch.nn.LeakyReLU)."
            ],
            'contexts': [
                ["torch.tensor creates a tensor from data. It can take lists, numpy arrays, or other tensors as input."],
                ["torch.nn.Linear applies a linear transformation. torch.nn.Conv2d applies a 2D convolution operation."],
                ["Neural networks in PyTorch are created by inheriting from nn.Module and implementing the forward method."],
                ["torch.optim.Adam optimizer parameters include lr, betas, eps, weight_decay, and amsgrad."],
                ["torch.nn.functional.relu applies the ReLU activation function element-wise to input tensors."],
                ["PyTorch uses automatic differentiation through autograd for backpropagation."],
                ["Models can be saved using torch.save() and loaded using torch.load()."],
                ["torch.tensor is a function, torch.Tensor is a class for creating tensors."],
                ["torch.nn.Dropout randomly zeros some elements during training to prevent overfitting."],
                ["PyTorch provides various activation functions like ReLU, Sigmoid, Tanh, and LeakyReLU."]
            ],
            'ground_truth': [
                "Use torch.tensor(data) to create a tensor from data like lists or arrays",
                "Linear is for fully connected layers, Conv2d is for convolutional layers",
                "Inherit from nn.Module, define layers in __init__, implement forward method",
                "Parameters: params, lr, betas, eps, weight_decay, amsgrad",
                "Apply ReLU activation using torch.nn.functional.relu(x)",
                "Automatic through autograd, use loss.backward() and optimizer.step()",
                "Save with torch.save(model.state_dict(), path), load with torch.load(path)",
                "torch.tensor is a function, torch.Tensor is a class",
                "Use torch.nn.Dropout(p) where p is the dropout probability",
                "ReLU, Sigmoid, Tanh, LeakyReLU are common activation functions"
            ]
        }
        
        dataset = Dataset.from_dict(sample_data)
        logger.info(f"Created sample dataset with {len(dataset)} samples")
        return dataset
    
    def generate_answers(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Generate answers for questions using the RAG pipeline."""
        results = []
        
        for i, question in enumerate(questions):
            try:
                logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
                
                # Process query through pipeline
                result = self.pipeline.process_query(question)
                
                # Extract relevant information
                answer_data = {
                    'answer': result['answer'],
                    'contexts': [doc['page_content'] for doc in result.get('retrieved_chunks', [])],
                    'confidence': result['confidence'],
                    'pipeline_metrics': result['pipeline_metrics'],
                    'routing': result['routing']
                }
                
                results.append(answer_data)
                
                # Add delay to avoid overwhelming the system
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing question {question}: {e}")
                results.append({
                    'answer': f"Error processing question: {str(e)}",
                    'contexts': [],
                    'confidence': 0.0,
                    'pipeline_metrics': {},
                    'routing': {}
                })
        
        return results
    
    def prepare_evaluation_data(self, dataset: Dataset, generated_results: List[Dict]) -> Dataset:
        """Prepare data for RAGAS evaluation."""
        evaluation_data = {
            'question': dataset['question'],
            'answer': [result['answer'] for result in generated_results],
            'contexts': [result['contexts'] for result in generated_results],
            'ground_truth': dataset['ground_truth']
        }
        
        return Dataset.from_dict(evaluation_data)
    
    def run_evaluation(self, dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Run comprehensive RAGAS evaluation."""
        try:
            # Load or use provided dataset
            if dataset is None:
                dataset = self.load_test_dataset()
            
            logger.info(f"Starting evaluation with {len(dataset)} samples")
            
            # Generate answers using the pipeline
            generated_results = self.generate_answers(dataset['question'])
            
            # Prepare data for RAGAS
            eval_dataset = self.prepare_evaluation_data(dataset, generated_results)
            
            # Run RAGAS evaluation
            logger.info("Running RAGAS evaluation...")
            result = evaluate(
                dataset=eval_dataset,
                metrics=self.metrics
            )
            
            # Convert results to pandas for easier analysis
            results_df = result.to_pandas()
            
            # Calculate summary statistics
            summary_stats = self.calculate_summary_stats(results_df, generated_results)
            
            # Save results
            self.save_evaluation_results(results_df, summary_stats, generated_results)
            
            logger.info("Evaluation completed successfully")
            
            return {
                'detailed_results': results_df,
                'summary_stats': summary_stats,
                'generated_results': generated_results,
                'metrics': {
                    'faithfulness': results_df['faithfulness'].mean(),
                    'answer_relevancy': results_df['answer_relevancy'].mean(),
                    'context_recall': results_df['context_recall'].mean(),
                    'answer_correctness': results_df['answer_correctness'].mean(),
                    'context_precision': results_df['context_precision'].mean()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            raise
    
    def calculate_summary_stats(self, results_df: pd.DataFrame, generated_results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics from evaluation results."""
        stats = {
            'total_questions': len(results_df),
            'average_confidence': sum(r['confidence'] for r in generated_results) / len(generated_results),
            'average_processing_time': sum(r['pipeline_metrics'].get('total_time', 0) for r in generated_results) / len(generated_results),
            'routing_distribution': {},
            'error_rate': 0
        }
        
        # Calculate routing distribution
        routing_counts = {}
        for result in generated_results:
            route_type = result.get('routing', {}).get('route_type', 'unknown')
            routing_counts[route_type] = routing_counts.get(route_type, 0) + 1
        
        stats['routing_distribution'] = routing_counts
        
        # Calculate error rate
        error_count = sum(1 for result in generated_results if result['confidence'] == 0.0)
        stats['error_rate'] = error_count / len(generated_results)
        
        # Calculate metric averages
        for metric in ['faithfulness', 'answer_relevancy', 'context_recall', 'answer_correctness', 'context_precision']:
            if metric in results_df.columns:
                stats[f'average_{metric}'] = results_df[metric].mean()
                stats[f'std_{metric}'] = results_df[metric].std()
        
        return stats
    
    def save_evaluation_results(self, results_df: pd.DataFrame, summary_stats: Dict, generated_results: List[Dict]):
        """Save evaluation results to files."""
        timestamp = int(time.time())
        
        # Save detailed results
        results_file = self.cache_dir / f"ragas_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Save summary stats
        stats_file = self.cache_dir / f"ragas_summary_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        # Save generated results
        generated_file = self.cache_dir / f"generated_results_{timestamp}.json"
        with open(generated_file, 'w') as f:
            json.dump(generated_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {self.cache_dir}")
    
    def compare_evaluations(self, result_files: List[str]) -> Dict[str, Any]:
        """Compare results from multiple evaluations."""
        comparison = {
            'evaluations': [],
            'improvements': {},
            'regressions': {}
        }
        
        for file_path in result_files:
            try:
                results_df = pd.read_csv(file_path)
                evaluation_summary = {
                    'file': file_path,
                    'timestamp': file_path.split('_')[-1].replace('.csv', ''),
                    'metrics': {}
                }
                
                for metric in ['faithfulness', 'answer_relevancy', 'context_recall', 'answer_correctness', 'context_precision']:
                    if metric in results_df.columns:
                        evaluation_summary['metrics'][metric] = results_df[metric].mean()
                
                comparison['evaluations'].append(evaluation_summary)
                
            except Exception as e:
                logger.error(f"Error reading evaluation file {file_path}: {e}")
        
        # Sort evaluations by timestamp
        comparison['evaluations'].sort(key=lambda x: x['timestamp'])
        
        # Calculate improvements/regressions if multiple evaluations
        if len(comparison['evaluations']) > 1:
            latest = comparison['evaluations'][-1]
            previous = comparison['evaluations'][-2]
            
            for metric in latest['metrics']:
                if metric in previous['metrics']:
                    change = latest['metrics'][metric] - previous['metrics'][metric]
                    if change > 0:
                        comparison['improvements'][metric] = change
                    else:
                        comparison['regressions'][metric] = abs(change)
        
        return comparison
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report."""
        report = []
        report.append("# PyTorch RAG Assistant - Evaluation Report")
        report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary metrics
        report.append("## Summary Metrics")
        metrics = results.get('metrics', {})
        for metric, score in metrics.items():
            report.append(f"- **{metric.replace('_', ' ').title()}**: {score:.3f}")
        
        report.append("")
        
        # Detailed statistics
        stats = results.get('summary_stats', {})
        report.append("## Detailed Statistics")
        report.append(f"- **Total Questions**: {stats.get('total_questions', 0)}")
        report.append(f"- **Average Confidence**: {stats.get('average_confidence', 0):.3f}")
        report.append(f"- **Average Processing Time**: {stats.get('average_processing_time', 0):.2f}s")
        report.append(f"- **Error Rate**: {stats.get('error_rate', 0):.1%}")
        
        report.append("")
        
        # Routing distribution
        routing = stats.get('routing_distribution', {})
        if routing:
            report.append("## Query Routing Distribution")
            for route, count in routing.items():
                percentage = (count / stats.get('total_questions', 1)) * 100
                report.append(f"- **{route}**: {count} ({percentage:.1f}%)")
        
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        
        if metrics.get('faithfulness', 0) < 0.8:
            report.append("- Improve answer faithfulness by enhancing context filtering")
        
        if metrics.get('answer_relevancy', 0) < 0.8:
            report.append("- Improve answer relevancy by better query understanding")
        
        if metrics.get('context_recall', 0) < 0.8:
            report.append("- Improve context recall by expanding retrieval scope")
        
        if metrics.get('answer_correctness', 0) < 0.8:
            report.append("- Improve answer correctness by enhancing generation prompts")
        
        if stats.get('error_rate', 0) > 0.1:
            report.append("- Reduce error rate by improving error handling and fallbacks")
        
        return "\n".join(report)


def main():
    """Main function for running evaluation."""
    evaluator = RAGASEvaluator()
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Generate and print report
    report = evaluator.generate_evaluation_report(results)
    print(report)
    
    # Save report
    report_file = Path(settings.evaluation_cache_dir) / f"evaluation_report_{int(time.time())}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
