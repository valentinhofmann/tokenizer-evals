import argparse
import os
import json

from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AddedToken

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def upload_tokenizer_to_hub(
    local_tokenizer_path,
    repository_owner,
    repository_name,
    commit_message=None,
    custom_tokens=None,
    max_length=8192,
):
    print(f"Loading tokenizer from: {local_tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)

    if custom_tokens and len(custom_tokens) > 0:
        # Convert all tokens to AddedToken objects except for 'additional_special_tokens'
        special_tokens_dict = {}
        for key, value in custom_tokens.items():
            if key == "additional_special_tokens":
                special_tokens_dict[key] = value
            else:
                special_tokens_dict[key] = AddedToken(**value, special=True)

        num_added = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added} custom tokens: {custom_tokens}")

    # Check if merges.txt exists
    merges_path = os.path.join(local_tokenizer_path, "merges.txt")
    has_merges = os.path.exists(merges_path)
    if has_merges:
        print(f"Found merges.txt file at: {merges_path}")

    # Set model max length
    tokenizer.model_max_length = max_length
    print(f"Set model max length to: {max_length}")

    # Set chat template
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    print("Set default chat template")

    # Create repository ID
    repo_id = f"{repository_owner}/{repository_name}"

    # Push the tokenizer to the hub
    tokenizer.push_to_hub(
        repo_id=repo_id,
        private=True,
        commit_message=f"{commit_message}",
    )

    # If merges.txt exists, add it to the repository
    if has_merges:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=merges_path,
            path_in_repo="merges.txt",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add merges.txt file",
        )
        print("Added merges.txt file to the repository")

    print(f"Successfully uploaded tokenizer to: {repo_id} (private repository)")


def main():
    parser = argparse.ArgumentParser(
        description="Upload a tokenizer to Hugging Face Hub"
    )
    parser.add_argument(
        "--path", required=True, help="Local directory path to the tokenizer"
    )
    parser.add_argument("--owner", required=True, help="Hugging Face repository owner")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument(
        "--message",
        help="Commit message for the upload",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--custom-tokens-file",
        help="Path to JSON file containing a list of custom tokens",
        required=True,
    )
    parser.add_argument(
        "--max-length",
        help="Set model max length (default: 8192)",
        type=int,
        default=8192,
    )

    args = parser.parse_args()

    # Load custom tokens from JSON file if provided
    custom_tokens = None
    if args.custom_tokens_file:
        try:
            with open(args.custom_tokens_file, "r") as f:
                custom_tokens = json.load(f)
            print(
                f"Loaded {len(custom_tokens)} custom tokens from {args.custom_tokens_file}"
            )
            print("Custom tokens:")
            for key, value in custom_tokens.items():
                if isinstance(value, list):
                    print(f"  {key}: {', '.join(repr(token) for token in value)}")
                else:
                    print(f"  {key}: {repr(value)}")
        except Exception as e:
            print(f"Error loading custom tokens file: {e}")
            return

    upload_tokenizer_to_hub(
        args.path,
        args.owner,
        args.repo,
        args.message,
        custom_tokens,
        args.max_length,
    )


if __name__ == "__main__":
    main()
