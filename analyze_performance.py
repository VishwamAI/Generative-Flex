import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_validation_results():
    """Load validation results from the latest training log"""
    log_dir = Path("logs")
    training_logs = sorted(log_dir.glob("training_*.log"), key=os.path.getmtime)

    if not training_logs:
        raise FileNotFoundError("No training logs found")

    latest_log = training_logs[-1]
    logger.info(f"Analyzing log file: {latest_log}")

    results = defaultdict(list)
    current_category = None

    with open(latest_log) as f:
        for line in f:
            if "Processing problem category:" in line:
                current_category = line.split(":")[-1].strip()
            elif "Batch accuracy:" in line and current_category:
                acc = float(line.split(":")[-1].strip())
                results[current_category].append(acc)

    return results

def analyze_performance():
    """Analyze model performance by problem category"""
    try:
        results = load_validation_results()

        # Calculate statistics per category
        stats = {}
        for category, accuracies in results.items():
            stats[category] = {
                'mean_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
                'num_samples': len(accuracies),
                'min_accuracy': min(accuracies) if accuracies else 0,
                'max_accuracy': max(accuracies) if accuracies else 0
            }

        # Create performance report
        report = ["Model Performance Analysis by Problem Category\n"]
        report.append("=" * 50 + "\n")

        for category, metrics in sorted(stats.items(), key=lambda x: x[1]['mean_accuracy'], reverse=True):
            report.append(f"\nCategory: {category}")
            report.append(f"Mean Accuracy: {metrics['mean_accuracy']:.2%}")
            report.append(f"Number of Samples: {metrics['num_samples']}")
            report.append(f"Range: {metrics['min_accuracy']:.2%} - {metrics['max_accuracy']:.2%}\n")

        # Save report
        report_path = "performance_analysis.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(report))
        logger.info(f"Performance report saved to {report_path}")

        # Create visualization
        plt.figure(figsize=(12, 6))
        categories = list(stats.keys())
        accuracies = [s['mean_accuracy'] for s in stats.values()]

        sns.barplot(x=accuracies, y=categories)
        plt.title("Model Performance by Problem Category")
        plt.xlabel("Accuracy")

        plt.tight_layout()
        plt.savefig("performance_by_category.png")
        logger.info("Performance visualization saved to performance_by_category.png")

        return stats

    except Exception as e:
        logger.error(f"Error analyzing performance: {str(e)}")
        return None

if __name__ == "__main__":
    analyze_performance()