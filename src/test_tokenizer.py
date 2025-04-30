import argparse
import os
import sys
import tempfile
from typing import Optional

import numpy as np
from transformers import AutoTokenizer

SAMPLE_TEXT = """
When implementing advanced algorithms, it's important to consider performance implications.
Here's a simple example in Python:

def load_and_test_tokenizer(tokenizer_name, npy_file_path, max_tokens=2048):
    try:
        # Load the tokenizer
        print(f"Loading tokenizer test_tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
"""


def load_and_test_tokenizer(
    tokenizer_name: str, npy_file_path: Optional[str], max_tokens: int = 2048
):
    try:
        # Load the tokenizer
        print(f"Loading tokenizer {tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        print(f"Tokenizer loaded: {tokenizer_name}")
        print(f"Tokenizer type: {type(tokenizer).__name__}")
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        print(f"Tokenizer model max length: {tokenizer.model_max_length}")
        print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")

        if npy_file_path is None:
            print("No file path provided. Tokenizing sample text...")
            token_ids = tokenizer.encode(SAMPLE_TEXT, add_special_tokens=False)

            # Create a temporary directory and file
            temp_dir = tempfile.mkdtemp(prefix="tokenizer_test_")
            temp_file = os.path.join(temp_dir, "test-data001.npy")

            print(f"Creating temporary token file at {temp_file}...")
            data_mmap = np.memmap(
                temp_file, mode="w+", dtype=np.uint32, shape=(len(token_ids),)
            )
            data_mmap[:] = token_ids
            data_mmap.flush()
            npy_file_path = temp_file

        print(f"Loading tokens from {npy_file_path}...")
        tokens = np.memmap(npy_file_path, dtype=np.uint32, mode="r")

        # Basic information
        print(f"\nToken file information:")
        print(f"Total tokens: {len(tokens)}")
        print(f"Token array shape: {tokens.shape}")
        print(f"Token array dtype: {tokens.dtype}")

        # Get subset of tokens
        tokens_to_show = tokens[:max_tokens]
        print(f"\nDecoding first {len(tokens_to_show)} tokens...")

        # Convert IDs to tokens first
        tokens_words = tokenizer.convert_ids_to_tokens(tokens_to_show.tolist())

        # For debugging - show a few raw tokens
        print("\nSample raw tokens (with special):")
        print(tokens_words[:10])

        # Decode properly
        decoded_text = tokenizer.decode(
            tokens_to_show, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        print("\nDecoded text:")
        print("=" * 100)
        print(decoded_text)
        print("=" * 100)

        # Print some token statistics
        print(f"\nToken statistics:")
        print(f"Unique tokens: {len(np.unique(tokens_to_show))}")
        print(f"Min token ID: {tokens_to_show.min()}")
        print(f"Max token ID: {tokens_to_show.max()}")

    except FileNotFoundError:
        print(f"Error: Could not find the file {npy_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test tokenizer with .npy token file")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Name of the tokenizer from HuggingFace Hub",
    )
    parser.add_argument(
        "--file",
        type=str,
        required=False,
        help="Path to the .npy file containing tokens",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to decode",
    )

    args = parser.parse_args()

    load_and_test_tokenizer(args.tokenizer, args.file, args.max_tokens)


if __name__ == "__main__":
    main()
