import os
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_mmmu_dataset():
    """Load the MMMU dataset."""
    try:
        dataset = load_dataset("MMMU/MMMU", name="Math")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def analyze_validation_set(dataset):
    """Analyze the validation set problems and their categories."""
    if not dataset or 'validation' not in dataset:
        logger.error("Dataset or validation split not available")
        return None

    validation_set = dataset['validation']

    # Category analysis
    categories = defaultdict(lambda: {'total': 0, 'correct': 0})

    # Extract validation metrics from logs
    validation_metrics = {}
    log_files = [f for f in os.listdir('logs') if f.startswith('training_')]
    if log_files:
        latest_log = sorted(log_files)[-1]
        with open(os.path.join('logs', latest_log), 'r') as f:
            for line in f:
                if 'Validation math accuracy:' in line:
                    try:
                        accuracy = float(line.split(':')[-1].strip())
                        validation_metrics['overall_accuracy'] = accuracy
                    except ValueError:
                        pass
                elif 'Validation loss:' in line:
                    try:
                        loss = float(line.split(':')[-1].strip())
                        if not isinstance(loss, complex):  # Filter out nan values
                            validation_metrics['validation_loss'] = loss
                    except ValueError:
                        pass

    # Analyze problems by category
    for example in validation_set:
        subfield = example.get('subfield', 'Unknown')
        topic_difficulty = example.get('topic_difficulty', 'Unknown')

        # Normalize subfield names
        if 'algebra' in subfield.lower():
            category = 'Algebra'
        elif 'calculus' in subfield.lower():
            category = 'Calculus'
        elif 'probability' in subfield.lower() or 'statistics' in subfield.lower():
            category = 'Probability & Statistics'
        elif 'geometry' in subfield.lower():
            category = 'Geometry'
        elif 'number' in subfield.lower():
            category = 'Number Theory'
        else:
            category = 'Other'

        categories[category]['total'] += 1
        categories[category]['difficulty'] = categories[category].get('difficulty', []) + [topic_difficulty]

    # Calculate statistics
    stats = {
        'overall': validation_metrics,
        'categories': {}
    }

    for category, data in categories.items():
        total = data['total']
        difficulties = data['difficulty']
        difficulty_distribution = defaultdict(int)
        for diff in difficulties:
            difficulty_distribution[diff] += 1

        stats['categories'][category] = {
            'total_problems': total,
            'percentage': (total / len(validation_set)) * 100,
            'difficulty_distribution': dict(difficulty_distribution)
        }

    return stats

def generate_visualization(stats):
    """Generate visualizations of the analysis."""
    if not stats:
        return

    # Category distribution plot
    plt.figure(figsize=(12, 6))
    categories = list(stats['categories'].keys())
    percentages = [data['percentage'] for data in stats['categories'].values()]

    sns.barplot(x=percentages, y=categories)
    plt.title('Distribution of Mathematical Categories in Validation Set')
    plt.xlabel('Percentage of Problems')
    plt.tight_layout()
    plt.savefig('category_distribution.png')
    plt.close()

def generate_report(stats):
    """Generate a comprehensive analysis report."""
    if not stats:
        logger.error("No statistics available for report generation")
        return

    report = ["MMMU Mathematical Categories Analysis\n"]
    report.append("=" * 50 + "\n")

    # Overall metrics
    if 'overall' in stats and stats['overall']:
        report.append("\nOverall Performance Metrics:")
        report.append("-" * 30)
        for metric, value in stats['overall'].items():
            report.append(f"{metric.replace('_', ' ').title()}: {value:.4f}")

    # Category breakdown
    report.append("\n\nCategory Distribution:")
    report.append("-" * 30)

    # Sort categories by percentage
    sorted_categories = sorted(
        stats['categories'].items(),
        key=lambda x: x[1]['percentage'],
        reverse=True
    )

    for category, data in sorted_categories:
        report.append(f"\n{category}:")
        report.append(f"  Total Problems: {data['total_problems']}")
        report.append(f"  Percentage: {data['percentage']:.2f}%")

        if 'difficulty_distribution' in data:
            report.append("  Difficulty Distribution:")
            for diff, count in data['difficulty_distribution'].items():
                report.append(f"    {diff}: {count} problems")

    # Save report
    report_path = "mmmu_category_analysis.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    logger.info(f"Category analysis report saved to {report_path}")

    # Save stats as JSON for further analysis
    with open('mmmu_category_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info("Category statistics saved to mmmu_category_stats.json")

def main():
    """Main analysis function."""
    dataset = load_mmmu_dataset()
    if dataset:
        stats = analyze_validation_set(dataset)
        if stats:
            generate_visualization(stats)
            generate_report(stats)

if __name__ == "__main__":
    main()