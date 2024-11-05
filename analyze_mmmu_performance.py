import os
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch
from src.data.mmmu_loader import MMUDataset
from src.models.enhanced_transformer import EnhancedTransformer
from src.config.config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_mmmu_dataset():
    """Load the MMMU validation dataset"""
    try:
        dataset = MMUDataset('Math', 'validation')
        logger.info(f"Successfully loaded MMMU validation dataset with {len(dataset)} examples")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def analyze_problem_categories(dataset):
    """Analyze and categorize problems in the dataset"""
    categories = defaultdict(list)

    try:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if isinstance(sample, dict):
                # Extract problem category/type
                category = sample.get('subject_name', 'Unknown')
                if 'algebra' in category.lower():
                    main_category = 'Algebra'
                elif 'calculus' in category.lower():
                    main_category = 'Calculus'
                elif 'probability' in category.lower() or 'statistics' in category.lower():
                    main_category = 'Probability & Statistics'
                elif 'geometry' in category.lower():
                    main_category = 'Geometry'
                elif 'number' in category.lower() or 'arithmetic' in category.lower():
                    main_category = 'Number Theory'
                else:
                    main_category = 'Other'

                categories[main_category].append(idx)

        return categories
    except Exception as e:
        logger.error(f"Error analyzing problem categories: {str(e)}")
        return None

def load_validation_results():
    """Load validation results from the most recent training log"""
    log_dir = Path("logs")
    training_logs = sorted(log_dir.glob("training_*.log"), key=os.path.getmtime)

    if not training_logs:
        logger.error("No training logs found")
        return None

    latest_log = training_logs[-1]
    logger.info(f"Analyzing log file: {latest_log}")

    results = {
        'overall_accuracy': None,
        'best_validation_loss': None,
        'category_performance': defaultdict(list)
    }

    try:
        with open(latest_log, 'r') as f:
            for line in f:
                if "Validation math accuracy:" in line:
                    try:
                        accuracy = float(line.split(":")[-1].strip())
                        results['overall_accuracy'] = accuracy
                    except ValueError:
                        continue
                elif "Best validation loss:" in line:
                    try:
                        loss = float(line.split(":")[-1].strip())
                        results['best_validation_loss'] = loss
                    except ValueError:
                        continue

        return results
    except Exception as e:
        logger.error(f"Error loading validation results: {str(e)}")
        return None

def generate_performance_report(categories, results):
    """Generate a comprehensive performance report"""
    if not results or not categories:
        logger.error("Missing results or categories data")
        return

    report = ["MMMU Mathematical Reasoning Performance Analysis\n"]
    report.append("=" * 50 + "\n")

    # Overall Performance
    if results['overall_accuracy'] is not None:
        report.append(f"\nOverall Mathematical Reasoning Accuracy: {results['overall_accuracy']:.2%}")
    if results['best_validation_loss'] is not None:
        report.append(f"Best Validation Loss: {results['best_validation_loss']:.4f}\n")

    # Category Distribution
    report.append("\nProblem Category Distribution:")
    report.append("-" * 30)
    total_problems = sum(len(probs) for probs in categories.values())

    for category, problems in sorted(categories.items()):
        count = len(problems)
        percentage = count / total_problems * 100
        report.append(f"\n{category}:")
        report.append(f"  Number of Problems: {count}")
        report.append(f"  Percentage of Dataset: {percentage:.1f}%")

    # Save report
    report_path = "mmmu_performance_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    logger.info(f"Performance report saved to {report_path}")

    # Generate visualization
    plt.figure(figsize=(12, 6))
    category_counts = [len(probs) for probs in categories.values()]
    category_names = list(categories.keys())

    sns.barplot(x=category_counts, y=category_names)
    plt.title("MMMU Problem Category Distribution")
    plt.xlabel("Number of Problems")
    plt.tight_layout()

    viz_path = "mmmu_category_distribution.png"
    plt.savefig(viz_path)
    logger.info(f"Category distribution visualization saved to {viz_path}")

def main():
    """Main analysis function"""
    # Load dataset
    dataset = load_mmmu_dataset()
    if not dataset:
        return

    # Analyze problem categories
    categories = analyze_problem_categories(dataset)
    if not categories:
        return

    # Load validation results
    results = load_validation_results()
    if not results:
        return

    # Generate comprehensive report
    generate_performance_report(categories, results)

if __name__ == "__main__":
    main()