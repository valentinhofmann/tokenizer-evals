import argparse
import logging

from transformers import AutoTokenizer

import utils

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--n_samples", type=int, required=False, default=1000)
    parser.add_argument(
        "--english_only",
        action="store_true",
        required=False,
    )
    parser.add_argument("--seed", type=int, required=False, default=42)
    parser.add_argument("--dataset", type=str, required=False, default=None)
    parser.add_argument(
        "--write_file",
        action="store_true",
        required=False,
        help="Write the results to a file per dataset",
    )
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    metric = "fertility"
    results = {}
    task_name = args.dataset if args.dataset else "all"

    if args.dataset:
        logger.info(f" Evaluating {args.tokenizer} on dataset: {args.dataset}")
        result = process_dataset(
            args.dataset, tokenizer, args.n_samples, args.english_only
        )
        results[args.dataset] = result
        utils.display_metric(args.dataset, result, metric)
    else:
        logger.info(
            f" Evaluating {args.tokenizer} on all datasets with up to {args.n_samples} samples"
        )

        for name in utils.ALL_DATASETS:
            result = process_dataset(name, tokenizer, args.n_samples, args.english_only)

            results[name] = result
            utils.display_metric(name, result, metric)

    if args.write_file:
        utils.write_json(task_name, results, metric, tokenizer)


def process_dataset(dataset, tokenizer, n_samples, english_only):
    data = utils.load_data(dataset, n_samples, english_only)
    result = utils.compute_metrics(data, tokenizer)

    return result


if __name__ == "__main__":
    main()
