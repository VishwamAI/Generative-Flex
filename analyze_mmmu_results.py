from collections import defaultdict
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt
import os
import seaborn as sns


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_validation_results(self):    """Parse validation results from training logs"""
        log_dir = Path("logs")
        training_logs = sorted(log_dir.glob("training_*.log"), key=os.path.getmtime)
        
        if not training_logs: logger.error("No training logs found")
        return None
        
        latest_log = training_logs[-1]
        logger.info(f"Analyzing log file: {latest_log}")
        
        # Initialize results dictionary
        results = {
        "overall_accuracy": None,
        "best_validation_loss": None,
        "problem_types": defaultdict(list),
        }
        
        current_problem = None
        
        with open(latest_log, "r") as f: forlinein, f:
        # Extract overall metrics
        if "Validation math accuracy:" in line: try: accuracy = float(line.split(":")[-1].strip())
        results["overall_accuracy"] = accuracy
        except ValueError: continueelif"Best validation loss:" in line: try: loss = float(line.split(":")[-1].strip())
        results["best_validation_loss"] = loss
        except ValueError: continue# Look for problem type indicators in the input text
        if "problem type:" in line.lower():
        problem_text = line.lower()
        if "algebra" in problem_text: current_problem = "Algebra"
        elif "calculus" in problem_text: current_problem= "Calculus"
        elif "probability" in problem_text or "statistics" in problem_text: current_problem= "Probability & Statistics"
        elif "geometry" in problem_text: current_problem= "Geometry"
        elif "number theory" in problem_text or "arithmetic" in problem_text: current_problem= "Number Theory", else: current_problem = "Other"
        
        # Look for accuracy metrics following problem type
        if current_problem and "correct:" in line.lower():
        try: correct = "true" in line.lower() or "1" in line.split()[-1]
        results["problem_types"][current_problem].append(correct)
        current_problem = None
        except Exception: continuereturnresults
        
        
                def generate_performance_report(results) -> None:                    """Generate a comprehensive performance report"""
        if not results: logger.error("No results data available")
            return
        
            report = ["MMMU Mathematical Reasoning Performance Analysis\n"]
            report.append("=" * 50 + "\n")
        
            # Overall Performance
            if results["overall_accuracy"] is not None: report.append(f"\nOverall Mathematical Reasoning Accuracy: {results['overall_accuracy']:.2%}")
                if results["best_validation_loss"] is not None: report.append(f"Best Validation Loss: {results['best_validation_loss']:.4f}\n")
        
                    # Performance by Category
                    report.append("\nPerformance by Problem Category:")
                    report.append("-" * 30)
        
                    category_metrics = {}
                    for category, outcomes in results["problem_types"].items():
                if outcomes: correct = sum(1 for x in outcomes if x)
                    total = len(outcomes)
                    accuracy = correct / total if total > 0 else 0
                    category_metrics[category] = {
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    }

                    # Sort categories by accuracy
                    for category, metrics in sorted(category_metrics.items(), key=lambda x: x[1]["accuracy"], reverse=True
                    ):
                        report.append(f"\n{category}:")
                        report.append(f"  Accuracy: {metrics['accuracy']:.2%}")
                        report.append(f"  Correct: {metrics['correct']}/{metrics['total']}")

                        # Save report
                        report_path = "mmmu_performance_report.txt"
                        with open(report_path, "w") as f: f.write("\n".join(report))
                            logger.info(f"Performance report saved to {report_path}")

                            # Generate visualization
                            if category_metrics: plt.figure(figsize=(12, 6))
                                categories = []
                                accuracies = []

                                for category, metrics in sorted(category_metrics.items(),
                                key=lambda x: x[1]["accuracy"],
                                reverse=True):
                                    categories.append(category)
                                    accuracies.append(metrics["accuracy"])

                                    sns.barplot(x=accuracies, y=categories)
                                    plt.title("MMMU Performance by Problem Category")
                                    plt.xlabel("Accuracy")
                                    plt.tight_layout()

                                    viz_path = "mmmu_performance_by_category.png"
                                    plt.savefig(viz_path)
                                    logger.info(f"Performance visualization saved to {viz_path}")


def main(self):    """Main analysis function"""
        results = parse_validation_results()
        if results: generate_performance_report(results)
        
        
        if __name__ == "__main__":
        main()
        