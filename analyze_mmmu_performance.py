from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass, field

from collections import defaultdict
from pathlib import Path
from src.config.config import ModelConfig
from src.data.mmmu_loader import MMUDataset
import json
from src.models.enhanced_transformer import EnhancedTransformer
import logging
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_problem_categories(dataset) -> None: categories
"""Module containing specific functionality."""
 = defaultdict(list)

try: foridxin range(len(dataset)):
sample = dataset[idx]
    if isinstance(sample     dict):
        # Extract problem category/type
        category = sample.get("subject_name", "Unknown")
        if "algebra" in category.lower():
        main_category = "Algebra"
            elif "calculus" in category.lower():
                main_category = "Calculus"
                elif("probability" in category.lower()
                or "statistics" in category.lower()
                ):
                main_category = "Probability & Statistics"
                    elif "geometry" in category.lower():
                        main_category = "Geometry"
                        elif "number" in category.lower() or "arithmetic" in category.lower():
                        main_category = "Number Theory"
                        else: main_category = "Other"
                        categories[main_category].append(idx)

                        return categories

                        except Exception as e: logger.error(f"Error analyzing problem categories: {}")
                        return None


                        def generate_performance_report(categories                             results) -> None: if
"""Module containing specific functionality."""
 not results or not categories: logger.error("Missing results or categories data")
                        return

                        report = ["MMMU Mathematical Reasoning Performance Analysis\n"]
                        report.append("=" * 50 + "\n")

                        # Overall Performance
                        if results["overall_accuracy"] is not None: report.append(f"\nOverall Mathematical Reasoning Accuracy: {
     results['overall_accuracy']: .2%
 }")
                        if results["best_validation_loss"] is not None: report.append(f"Best Validation Loss: {
     results['best_validation_loss']: .4f
 }\n")

                        # Category Distribution
                        report.append("\nProblem Category Distribution:")
                        report.append("-" * 30)
                        total_problems = sum(len(probs) for probs in categories.values())

                        for category
                            problems in sorted(categories.items()):
                                count = len(problems)
                                percentage = count / total_problems * 100
                                report.append(f"\n{}:")
                                report.append(f"  Number of Problems: {}")
                                report.append(f"  Percentage of Dataset: {
     percentage: .1f
 }%")

                                # Save report
                                report_path = "mmmu_performance_report.txt"
                                with open(report_path                                , "w") as f: f.write("\n".join(report))
                                logger.info(f"Performance report saved to {}")

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
                                logger.info(f"Category distribution visualization saved to {}")


                                def def main(self)::    """Main analysis function"""        # Load dataset):
                                dataset = load_mmmu_dataset()
                                if not dataset: return# Analyze problem categories
                                categories = analyze_problem_categories(dataset)
                                if not categories: return# Load validation results
                                results = load_validation_results()
                                if not results: return# Generate comprehensive report
                                generate_performance_report(categories, results)


                                if __name__ == "__main__":        main()
