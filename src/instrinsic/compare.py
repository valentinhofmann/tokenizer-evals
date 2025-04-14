import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text


def read_json_files(directory):
    """Read JSON files in the directory and organize by tokenizer and dataset."""
    tokenizer_results = {}
    directory_path = Path(directory)

    for file_path in directory_path.glob("*.json"):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                # Extract tokenizer name from filename
                tokenizer_name = file_path.stem.split("_eval_")[0]

                # If this is a multi-dataset file with the expected structure
                if isinstance(data, dict) and any(
                    isinstance(v, dict) and "vocab_coverage" in v for v in data.values()
                ):
                    tokenizer_results[tokenizer_name] = data
                # If it's a single tokenizer file with the old format
                elif len(data) == 1 and isinstance(list(data.values())[0], dict):
                    tokenizer_name = list(data.keys())[0]
                    tokenizer_results[tokenizer_name] = list(data.values())[0]
                else:
                    # Single dataset file
                    tokenizer_results[tokenizer_name] = data
            except json.JSONDecodeError:
                print(f"Error parsing JSON from {file_path}")

    return tokenizer_results


def format_value(value):
    """Format a value appropriately for display with fixed 2 decimal precision."""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def create_dataset_table(data, dataset):
    """Create a rich table comparing metrics across tokenizers for a specific dataset."""
    console = Console()

    table = Table(title=f"Tokenizer Comparison for {dataset}")
    table.add_column("Metric", style="cyan")

    # Get tokenizers that have this dataset
    tokenizers = [t for t in data.keys() if dataset in data[t]]

    # Add columns for each tokenizer
    for tokenizer in sorted(tokenizers):
        table.add_column(tokenizer, justify="right")

    # Get all metrics for this dataset
    all_metrics = set()
    for tokenizer in tokenizers:
        for metric, value in data[tokenizer][dataset].items():
            if isinstance(value, dict):
                for sub_metric in value.keys():
                    all_metrics.add(f"{metric}.{sub_metric}")
            else:
                all_metrics.add(metric)

    # Add rows for each metric
    for metric in sorted(all_metrics):
        row = [metric]
        metric_values = {}

        # Collect values for this metric across tokenizers
        for tokenizer in sorted(tokenizers):
            parent_metric, *sub_parts = metric.split(".", 1)
            if parent_metric in data[tokenizer][dataset]:
                if sub_parts:
                    # Handle nested metrics
                    sub_metric = sub_parts[0]
                    if sub_metric in data[tokenizer][dataset][parent_metric]:
                        value = data[tokenizer][dataset][parent_metric][sub_metric]
                        metric_values[tokenizer] = value
                else:
                    # Handle top-level metrics
                    value = data[tokenizer][dataset][parent_metric]
                    metric_values[tokenizer] = value

        # Find max value for highlighting
        numeric_values = [
            v for v in metric_values.values() if isinstance(v, (int, float))
        ]
        max_value = max(numeric_values) if numeric_values else None

        # Add formatted values to the row
        for tokenizer in sorted(tokenizers):
            if tokenizer in metric_values:
                value = metric_values[tokenizer]
                value_str = format_value(value)

                # Bold maximum values
                if isinstance(value, (int, float)) and value == max_value:
                    row.append(Text(value_str, style="bold green"))
                else:
                    row.append(value_str)
            else:
                row.append("")

        table.add_row(*row)

    console.print(table)
    console.print()  # Add blank line between tables


def create_summary_table(data):
    """Create a summary table showing average metrics across all datasets for each tokenizer."""
    console = Console()

    table = Table(title="Summary: Average Metrics Across All Datasets")
    table.add_column("Metric", style="cyan")

    # Get all tokenizers
    all_tokenizers = sorted(data.keys())

    # Add columns for each tokenizer
    for tokenizer in all_tokenizers:
        table.add_column(tokenizer, justify="right")

    # Get all metrics across all datasets and tokenizers
    all_metrics = set()

    for tokenizer, datasets in data.items():
        for dataset, metrics in datasets.items():
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    for sub_metric in value.keys():
                        all_metrics.add(f"{metric}.{sub_metric}")
                else:
                    all_metrics.add(metric)

    # Calculate averages for each metric for each tokenizer
    averages = {}

    for metric in all_metrics:
        averages[metric] = {}

        for tokenizer in all_tokenizers:
            values = []

            for dataset, metrics in data[tokenizer].items():
                parent_metric, *sub_parts = metric.split(".", 1)

                if parent_metric in metrics:
                    if sub_parts:
                        # Handle nested metrics
                        sub_metric = sub_parts[0]
                        if sub_metric in metrics[parent_metric]:
                            value = metrics[parent_metric][sub_metric]
                            if isinstance(value, (int, float)):
                                values.append(value)
                    else:
                        # Handle top-level metrics
                        value = metrics[parent_metric]
                        if isinstance(value, (int, float)):
                            values.append(value)

            if values:
                averages[metric][tokenizer] = sum(values) / len(values)

    # Add rows for each metric
    for metric in sorted(all_metrics):
        row = [metric]
        metric_values = averages[metric]

        # Find max value for highlighting
        numeric_values = [
            v for v in metric_values.values() if isinstance(v, (int, float))
        ]
        max_value = max(numeric_values) if numeric_values else None

        # Add formatted values to the row
        for tokenizer in all_tokenizers:
            if tokenizer in metric_values:
                value = metric_values[tokenizer]
                value_str = format_value(value)

                # Bold maximum values
                if isinstance(value, (int, float)) and value == max_value:
                    row.append(Text(value_str, style="bold green"))
                else:
                    row.append(value_str)
            else:
                row.append("N/A")

        table.add_row(*row)

    console.print(table)
    console.print()  # Add blank line


def compare_tokenizers(data, dataset_filter=None):
    """Compare tokenizers across datasets, creating one table per dataset."""
    # Get all datasets
    all_datasets = set()
    for tokenizer, datasets in data.items():
        all_datasets.update(datasets.keys())

    if dataset_filter:
        # Only create a table for the specified dataset
        if dataset_filter in all_datasets:
            create_dataset_table(data, dataset_filter)
        else:
            console = Console()
            console.print(
                f"[bold red]Dataset '{dataset_filter}' not found in data[/bold red]"
            )
    else:
        # First create a summary table with averages across datasets
        create_summary_table(data)

        # Then create a table for each dataset
        for dataset in sorted(all_datasets):
            create_dataset_table(data, dataset)


def main():
    parser = argparse.ArgumentParser(
        description="Compare tokenizer evaluation results across datasets and tokenizers."
    )
    parser.add_argument("--dir", type=str, help="Directory containing JSON files")
    parser.add_argument(
        "--file", type=str, help="Single JSON file with evaluation results"
    )
    parser.add_argument(
        "--dataset", type=str, help="Filter results to a specific dataset"
    )
    args = parser.parse_args()

    data = {}

    if args.file:
        # Read from a single file
        try:
            with open(args.file, "r") as f:
                file_data = json.load(f)
                # Extract tokenizer name from filename
                tokenizer_name = Path(args.file).stem.split("_eval_")[0]
                data = {tokenizer_name: file_data}
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading file {args.file}: {e}")
            return
    elif args.dir:
        # Read from a directory
        data = read_json_files(args.dir)
    else:
        # Default to .output directory if it exists
        output_dir = Path(".output")
        if output_dir.exists():
            data = read_json_files(output_dir)
        else:
            print("No file or directory specified, and no .output directory found.")
            return

    compare_tokenizers(data, args.dataset)


if __name__ == "__main__":
    main()
