import os
import re
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_problem_category(text):
    """Extract mathematical category from problem text."""
    text = text.lower()
    categories = {
        "Algebra": [
            "algebra",
            "equation",
            "polynomial",
            "variable",
            "solve for",
            "linear",
        ],
        "Calculus": [
            "calculus",
            "derivative",
            "integral",
            "differentiate",
            "integrate",
            "limit",
        ],
        "Probability & Statistics": [
            "probability",
            "statistics",
            "random",
            "distribution",
            "expected value",
        ],
        "Geometry": [
            "geometry",
            "triangle",
            "circle",
            "angle",
            "polygon",
            "area",
        ],
        "Number Theory": [
            "number theory",
            "prime",
            "factor",
            "divisor",
            "gcd",
            "lcm",
        ],
    }

    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "Other"


def parse_validation_outputs():
    """Parse validation outputs from the training logs."""
    log_dir = Path("logs")
    training_logs = sorted(
        log_dir.glob("training_*.log"), key=os.path.getmtime
    )

    if not training_logs:
        logger.error("No training logs found")
        return None

    latest_log = training_logs[-1]
    logger.info(f"Analyzing log file: {latest_log}")

    results = {
        "overall_accuracy": None,
        "best_validation_loss": None,
        "categories": defaultdict(lambda: {"correct": 0, "total": 0}),
    }

    current_problem = None
    current_category = None

    with open(latest_log, "r") as f:
        content = f.read()

        # Extract overall metrics
        accuracy_matches = re.findall(
            r"Validation math accuracy: ([\d.]+)", content
        )
        if accuracy_matches:
            results["overall_accuracy"] = float(accuracy_matches[-1])

        loss_matches = re.findall(r"Validation loss: ([\d.]+)", content)
        if loss_matches:
            results["best_validation_loss"] = float(loss_matches[-1])

        # Extract problem-specific information
        problem_blocks = re.split(r"Processing validation example", content)
        for block in problem_blocks[
            1:
        ]:  # Skip the first split as it's before any problem
            # Try to extract problem text
            problem_text = re.search(r"Input text: (.+?)(?=\n|$)", block)
            if problem_text:
                category = extract_problem_category(problem_text.group(1))
                results["categories"][category]["total"] += 1

                # Check if the answer was correct
                if "Correct answer" in block or "Answer matches" in block:
                    results["categories"][category]["correct"] += 1

    return results


def generate_performance_report(results):
    """Generate a comprehensive performance report."""
    if not results:
        logger.error("No results data available")
        return

    report = ["MMMU Mathematical Reasoning Performance Analysis\n"]
    report.append("=" * 50 + "\n")

    # Overall Performance
    if results["overall_accuracy"] is not None:
        report.append(
            f"\nOverall Mathematical Reasoning Accuracy: {results['overall_accuracy']:.2%}"
        )
    if results["best_validation_loss"] is not None:
        report.append(
            f"Best Validation Loss: {results['best_validation_loss']:.4f}\n"
        )

    # Category-specific Performance
    report.append("\nPerformance by Mathematical Category:")
    report.append("-" * 30)

    category_metrics = {}
    for category, metrics in results["categories"].items():
        if metrics["total"] > 0:
            accuracy = metrics["correct"] / metrics["total"]
            category_metrics[category] = {
                "accuracy": accuracy,
                "correct": metrics["correct"],
                "total": metrics["total"],
            }

    # Sort categories by accuracy
    for category, metrics in sorted(
        category_metrics.items(), key=lambda x: x[1]["accuracy"], reverse=True
    ):
        report.append(f"\n{category}:")
        report.append(f"  Accuracy: {metrics['accuracy']:.2%}")
        report.append(f"  Correct: {metrics['correct']}/{metrics['total']}")

    # Analysis of Strengths and Weaknesses
    report.append("\n\nModel Analysis:")
    report.append("-" * 30)

    # Identify top performing categories
    top_categories = sorted(
        category_metrics.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )
    if top_categories:
        report.append("\nStrengths:")
        for category, metrics in top_categories[:2]:
            if (
                metrics["accuracy"] > 0.5
            ):  # Only include if accuracy is above 50%
                report.append(
                    f"- {category}: {metrics['accuracy']:.2%} accuracy"
                )

        report.append("\nAreas for Improvement:")
        for category, metrics in top_categories[-2:]:
            if metrics["accuracy"] < 0.8:  # Include if accuracy is below 80%
                report.append(
                    f"- {category}: {metrics['accuracy']:.2%} accuracy"
                )

    # Save report
    report_path = "mmmu_detailed_performance.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    logger.info(f"Detailed performance report saved to {report_path}")

    # Generate visualization
    if category_metrics:
        plt.figure(figsize=(12, 6))
        categories = []
        accuracies = []

        for category, metrics in sorted(
            category_metrics.items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True,
        ):
            categories.append(category)
            accuracies.append(metrics["accuracy"])

        sns.barplot(x=accuracies, y=categories)
        plt.title("MMMU Performance by Mathematical Category")
        plt.xlabel("Accuracy")
        plt.tight_layout()

        viz_path = "mmmu_category_performance.png"
        plt.savefig(viz_path)
        logger.info(f"Performance visualization saved to {viz_path}")


def main():
    """Main analysis function."""
    results = parse_validation_outputs()
    if results:
        generate_performance_report(results)


if __name__ == "__main__":
    main()
