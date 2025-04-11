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

    if args.dataset:
        logger.info(f" Evaluating {args.tokenizer} on dataset: {args.dataset}")
        process_dataset(
            args.dataset, tokenizer, args.n_samples, args.english_only, args.write_file
        )
    else:
        logger.info(
            f" Evaluating {args.tokenizer} on all datasets with up to {args.n_samples} samples"
        )
        for name in utils.ALL_DATASETS:
            process_dataset(
                name, tokenizer, args.n_samples, args.english_only, args.write_file
            )


def process_dataset(dataset, tokenizer, n_samples, english_only, write_file=False):
    data = utils.load_data(dataset, n_samples, english_only)
    result = utils.compute_metrics(data, tokenizer)
    metric = "fertility"

    if write_file:
        utils.write_json(dataset, result, metric, tokenizer)

    utils.display_metric(dataset, result, metric)


if __name__ == "__main__":
    main()
