import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

# Define which metrics are better when higher or lower
# True: higher is better (positive delta = green)
# False: lower is better (negative delta = green)
METRIC_POLARITY = {
    "vocab_coverage": None,  # Coverage is neutral
    "compression_ratio": True,  # Higher compression is better
    "avg_tokens_per_word": False,  # Lower tokens per word is more efficient
    "avg_tokens_per_char": False,  # Lower tokens per char is more efficient
    "oov_rate": False,  # Lower out-of-vocab rate is better
    "subword_stats.subword_ratio": None,  # Neutral, depends on use case
    "subword_stats.full_word_ratio": None,  # Neutral, depends on use case
    "distribution_stats.unique_token_ratio": True,  # Higher diversity is better
    "distribution_stats.tokens.shannon.entropy": True,  # Higher entropy is better
    "distribution_stats.tokens.renyi.entropy": True,  # Higher entropy is better
    "distribution_stats.top_10_token_ratio": False,  # Lower concentration is better
}


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


def create_dataset_table(data, dataset, reference_tokenizer=None):
    """Create a rich table comparing metrics across tokenizers for a specific dataset."""
    console = Console()

    # Get tokenizers that have this dataset
    tokenizers = [t for t in data.keys() if dataset in data[t]]

    # Extract uniseg.renyi.entropy from any tokenizer (it's a dataset property)
    uniseg_entropy = None
    for tokenizer in tokenizers:
        if (
            "distribution_stats" in data[tokenizer][dataset]
            and "uniseg.renyi.entropy" in data[tokenizer][dataset]["distribution_stats"]
        ):
            uniseg_entropy = data[tokenizer][dataset]["distribution_stats"][
                "uniseg.renyi.entropy"
            ]
            break

    # Create table with title that includes uniseg entropy if available
    if uniseg_entropy is not None:
        table = Table(
            title=f"Tokenizer Comparison for {dataset} [Dataset Uniseg Renyi Entropy: {format_value(uniseg_entropy)}]"
        )
    else:
        table = Table(title=f"Tokenizer Comparison for {dataset}")

    table.add_column("Metric", style="cyan")

    # Check if reference tokenizer exists
    reference_exists = reference_tokenizer in tokenizers
    if reference_tokenizer and not reference_exists:
        console.print(
            f"[bold yellow]Warning: Reference tokenizer '{reference_tokenizer}' not found for dataset '{dataset}'[/bold yellow]"
        )

    # Add columns for each tokenizer
    for tokenizer in sorted(tokenizers):
        table.add_column(tokenizer, justify="right")

    # Add delta columns if reference tokenizer exists
    if reference_exists:
        table.add_column(
            f"Abs Δ vs {reference_tokenizer}", justify="right", style="magenta"
        )
        table.add_column(
            f"Rel Δ % vs {reference_tokenizer}", justify="right", style="magenta"
        )

    # Get all metrics for this dataset (excluding uniseg.renyi.entropy)
    all_metrics = set()
    for tokenizer in tokenizers:
        for metric, value in data[tokenizer][dataset].items():
            if isinstance(value, dict):
                for sub_metric in value.keys():
                    # Skip uniseg.renyi.entropy
                    if not (
                        metric == "distribution_stats"
                        and sub_metric == "uniseg.renyi.entropy"
                    ):
                        all_metrics.add(f"{metric}.{sub_metric}")
            else:
                all_metrics.add(metric)

    # Add rows for each metric
    for metric in sorted(all_metrics):
        # Skip uniseg.renyi.entropy (redundant check)
        if metric == "distribution_stats.uniseg.renyi.entropy":
            continue

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

        # Find best value for highlighting based on metric polarity
        numeric_values = [
            v for v in metric_values.values() if isinstance(v, (int, float))
        ]

        # Determine if higher or lower is better for this metric
        higher_is_better = None
        for metric_key, polarity in METRIC_POLARITY.items():
            if metric_key == metric or metric.endswith("." + metric_key.split(".")[-1]):
                higher_is_better = polarity
                break

        # Find the best value according to the metric's polarity
        best_value = None
        if numeric_values:
            if higher_is_better is True:  # Higher is better
                best_value = max(numeric_values)
            elif higher_is_better is False:  # Lower is better
                best_value = min(numeric_values)
            else:  # Neutral - don't highlight any value
                best_value = None

        # Add formatted values to the row
        for tokenizer in sorted(tokenizers):
            if tokenizer in metric_values:
                value = metric_values[tokenizer]
                value_str = format_value(value)

                # Bold best values based on the polarity
                if (
                    isinstance(value, (int, float))
                    and best_value is not None
                    and value == best_value
                ):
                    row.append(Text(value_str, style="bold green"))
                else:
                    row.append(value_str)
            else:
                row.append("")

        # Add delta columns if reference tokenizer exists
        if reference_exists and reference_tokenizer in metric_values:
            ref_value = metric_values[reference_tokenizer]

            # Calculate deltas against the next best value (not average)
            if isinstance(ref_value, (int, float)):
                # Check polarity mapping to determine which value to compare against
                higher_is_better = None
                for metric_key, polarity in METRIC_POLARITY.items():
                    if metric_key == metric or metric.endswith(
                        "." + metric_key.split(".")[-1]
                    ):
                        higher_is_better = polarity
                        break

                # Find the best non-reference value based on polarity
                best_non_ref_value = None
                for tokenizer in sorted(tokenizers):
                    if tokenizer != reference_tokenizer and tokenizer in metric_values:
                        val = metric_values[tokenizer]
                        if isinstance(val, (int, float)):
                            if best_non_ref_value is None:
                                best_non_ref_value = val
                            elif higher_is_better is True and val > best_non_ref_value:
                                best_non_ref_value = (
                                    val  # Higher is better, find highest
                                )
                            elif higher_is_better is False and val < best_non_ref_value:
                                best_non_ref_value = val  # Lower is better, find lowest
                            elif higher_is_better is None:
                                # For neutral metrics, find closest to reference
                                if abs(val - ref_value) < abs(
                                    best_non_ref_value - ref_value
                                ):
                                    best_non_ref_value = val

                if best_non_ref_value is not None:
                    # Calculate delta against next best value
                    abs_delta = ref_value - best_non_ref_value
                    abs_delta_str = format_value(abs_delta)

                    # Add plus sign for positive deltas
                    if abs_delta > 0:
                        abs_delta_str = "+" + abs_delta_str

                    # Set color based on polarity
                    if higher_is_better is True:  # Higher is better
                        delta_style = (
                            "green" if abs_delta > 0 else "red" if abs_delta < 0 else ""
                        )
                    elif higher_is_better is False:  # Lower is better
                        delta_style = (
                            "red" if abs_delta > 0 else "green" if abs_delta < 0 else ""
                        )
                    else:  # Neutral/Unknown
                        delta_style = "yellow" if abs_delta != 0 else ""

                    row.append(Text(abs_delta_str, style=delta_style))

                    # Calculate relative delta if best_non_ref_value is not zero
                    if best_non_ref_value != 0:
                        rel_delta = (abs_delta / best_non_ref_value) * 100
                        rel_delta_str = format_value(rel_delta)

                        # Add plus sign and percentage
                        if rel_delta > 0:
                            rel_delta_str = "+" + rel_delta_str
                        rel_delta_str += "%"

                        # Use same color as absolute delta
                        rel_delta_style = delta_style
                        row.append(Text(rel_delta_str, style=rel_delta_style))
                    else:
                        row.append("N/A")
                else:
                    row.append("N/A")
                    row.append("N/A")
            else:
                # Non-numeric reference value
                row.append("N/A")
                row.append("N/A")
        elif reference_exists:
            # Reference tokenizer exists but no value for this metric
            row.append("N/A")
            row.append("N/A")

        table.add_row(*row)

    console.print(table)
    console.print()  # Add blank line between tables


def create_summary_table(data, reference_tokenizer=None):
    """Create a summary table showing average metrics across all datasets for each tokenizer."""
    console = Console()

    table = Table(title="Summary: Average Metrics Across All Datasets")
    table.add_column("Metric", style="cyan")

    # Get all tokenizers
    all_tokenizers = sorted(data.keys())

    # Check if reference tokenizer exists
    reference_exists = reference_tokenizer in all_tokenizers
    if reference_tokenizer and not reference_exists:
        console.print(
            f"[bold yellow]Warning: Reference tokenizer '{reference_tokenizer}' not found[/bold yellow]"
        )

    # Add columns for each tokenizer
    for tokenizer in all_tokenizers:
        table.add_column(tokenizer, justify="right")

    # Add delta columns if reference tokenizer exists
    if reference_exists:
        table.add_column(
            f"Abs Δ vs {reference_tokenizer}", justify="right", style="magenta"
        )
        table.add_column(
            f"Rel Δ % vs {reference_tokenizer}", justify="right", style="magenta"
        )

    # Get all metrics across all datasets and tokenizers (excluding uniseg.renyi.entropy)
    all_metrics = set()
    for tokenizer, datasets in data.items():
        for dataset, metrics in datasets.items():
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    for sub_metric in value.keys():
                        # Skip uniseg.renyi.entropy
                        if not (
                            metric == "distribution_stats"
                            and sub_metric == "uniseg.renyi.entropy"
                        ):
                            all_metrics.add(f"{metric}.{sub_metric}")
                else:
                    all_metrics.add(metric)

    # Calculate averages for each metric for each tokenizer
    averages = {}

    for metric in all_metrics:
        averages[metric] = {}

        for tokenizer in all_tokenizers:
            values = []

            for _, metrics in data[tokenizer].items():
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

        # Add delta columns if reference tokenizer exists
        if reference_exists and reference_tokenizer in metric_values:
            ref_value = metric_values[reference_tokenizer]

            # Calculate deltas against the next best value (not average)
            if isinstance(ref_value, (int, float)):
                # Check polarity mapping to determine which value to compare against
                higher_is_better = None
                for metric_key, polarity in METRIC_POLARITY.items():
                    if metric_key == metric or metric.endswith(
                        "." + metric_key.split(".")[-1]
                    ):
                        higher_is_better = polarity
                        break

                # Find the best non-reference value based on polarity
                best_non_ref_value = None
                for tokenizer in all_tokenizers:
                    if tokenizer != reference_tokenizer and tokenizer in metric_values:
                        val = metric_values[tokenizer]
                        if isinstance(val, (int, float)):
                            if best_non_ref_value is None:
                                best_non_ref_value = val
                            elif higher_is_better is True and val > best_non_ref_value:
                                best_non_ref_value = (
                                    val  # Higher is better, find highest
                                )
                            elif higher_is_better is False and val < best_non_ref_value:
                                best_non_ref_value = val  # Lower is better, find lowest
                            elif higher_is_better is None:
                                # For neutral metrics, find closest to reference
                                if abs(val - ref_value) < abs(
                                    best_non_ref_value - ref_value
                                ):
                                    best_non_ref_value = val

                if best_non_ref_value is not None:
                    # Calculate delta against next best value
                    abs_delta = ref_value - best_non_ref_value
                    abs_delta_str = format_value(abs_delta)

                    # Add plus sign for positive deltas
                    if abs_delta > 0:
                        abs_delta_str = "+" + abs_delta_str

                    # Set color based on polarity
                    if higher_is_better is True:  # Higher is better
                        delta_style = (
                            "green" if abs_delta > 0 else "red" if abs_delta < 0 else ""
                        )
                    elif higher_is_better is False:  # Lower is better
                        delta_style = (
                            "red" if abs_delta > 0 else "green" if abs_delta < 0 else ""
                        )
                    else:  # Neutral/Unknown
                        delta_style = "yellow" if abs_delta != 0 else ""

                    row.append(Text(abs_delta_str, style=delta_style))

                    # Calculate relative delta if best_non_ref_value is not zero
                    if best_non_ref_value != 0:
                        rel_delta = (abs_delta / best_non_ref_value) * 100
                        rel_delta_str = format_value(rel_delta)

                        # Add plus sign and percentage
                        if rel_delta > 0:
                            rel_delta_str = "+" + rel_delta_str
                        rel_delta_str += "%"

                        # Use same color as absolute delta
                        rel_delta_style = delta_style
                        row.append(Text(rel_delta_str, style=rel_delta_style))
                    else:
                        row.append("N/A")
                else:
                    row.append("N/A")
                    row.append("N/A")
            else:
                # Non-numeric reference value
                row.append("N/A")
                row.append("N/A")
        elif reference_exists:
            # Reference tokenizer exists but no value for this metric
            row.append("N/A")
            row.append("N/A")

        table.add_row(*row)

    console.print(table)
    console.print()  # Add blank line


def compare_tokenizers(data, dataset_filter=None, reference_tokenizer=None):
    """Compare tokenizers across datasets, creating one table per dataset."""
    # Get all datasets
    all_datasets = set()
    for _, datasets in data.items():
        all_datasets.update(datasets.keys())

    if dataset_filter:
        # Only create a table for the specified dataset
        if dataset_filter in all_datasets:
            create_dataset_table(data, dataset_filter, reference_tokenizer)
        else:
            console = Console()
            console.print(
                f"[bold red]Dataset '{dataset_filter}' not found in data[/bold red]"
            )
    else:
        # First create a summary table with averages across datasets
        create_summary_table(data, reference_tokenizer)

        # Then create a table for each dataset
        for dataset in sorted(all_datasets):
            create_dataset_table(data, dataset, reference_tokenizer)


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
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Reference tokenizer to compare others against",
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

    compare_tokenizers(data, args.dataset, args.reference)


if __name__ == "__main__":
    main()
