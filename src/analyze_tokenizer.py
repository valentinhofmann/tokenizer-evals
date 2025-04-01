import argparse

from transformers import AutoTokenizer

import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        required=False,
        default=1000
    )
    parser.add_argument(
        "--english_only", 
        action="store_true",
        required=False,
    )
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    for dataset_name in utils.CHAT_DATASETS:

        # Load dataset
        data = utils.load_data(
            dataset_name, 
            args.n_samples, 
            args.english_only
        )

        # Compute metrics
        data = utils.compute_metrics(data, tokenizer)

        # Display metrics
        utils.display_metric(dataset_name, data, "fertility")


if __name__ == "__main__":
    main()
