import argparse
import os
import json

from huggingface_hub import HfApi
from transformers import AutoTokenizer, AddedToken

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def add_pretokenization_split(tokenizer):
    """
    Adds the pretokenization split pattern to the tokenizer configuration.
    """
    # Get the current tokenizer config
    config = tokenizer.pretrained_init_configuration

    # Add or update the pretokenization split pattern
    pretokenization_split = {
        "type": "Split",
        "pattern": {
            "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        },
        "behavior": "Removed",
        "invert": True,
    }

    # Update the config to include the pretokenization split
    if "pretokenizer" not in config:
        config["pretokenizer"] = {
            "type": "Sequence",
            "pretokenizers": [pretokenization_split],
        }
    elif config["pretokenizer"]["type"] == "Sequence":
        config["pretokenizer"]["pretokenizers"].append(pretokenization_split)
    else:
        # If there's an existing non-sequence pretokenizer, wrap it in a sequence
        existing = config["pretokenizer"]
        config["pretokenizer"] = {
            "type": "Sequence",
            "pretokenizers": [existing, pretokenization_split],
        }

    print("Added pretokenization split pattern to tokenizer configuration")
    return config


def upload_tokenizer_to_hub(
    local_tokenizer_path,
    repository_owner,
    repository_name,
    commit_message=None,
    custom_tokens=None,
    max_length=8192,
    add_pretokenization_split=True,
):
    print(f"Loading tokenizer from: {local_tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)

    if custom_tokens and len(custom_tokens) > 0:
        # Convert all tokens to AddedToken objects except for 'additional_special_tokens'
        special_tokens_dict = {}
        for key, value in custom_tokens.items():
            if key == "additional_special_tokens":
                # Handle additional_special_tokens as a list of strings
                if isinstance(value, list):
                    special_tokens_dict[key] = [
                        AddedToken(token, special=False) for token in value
                    ]
                else:
                    print(
                        f"Warning: 'additional_special_tokens' should be a list, got {type(value)}"
                    )
                    continue
            else:
                special_tokens_dict[key] = AddedToken(**value, special=False)

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

    if add_pretokenization_split:
        # Add pretokenization split pattern
        tokenizer.pretrained_init_configuration = add_pretokenization_split(tokenizer)

    # Create repository ID
    repo_id = f"{repository_owner}/{repository_name}"

    # Push the tokenizer to the hub
    tokenizer.push_to_hub(
        repo_id=repo_id,
        private=True,
        commit_message=f"{commit_message}",
    )

    # Upload all files from the tokenizer directory
    api = HfApi()
    print(f"Uploading all files from {local_tokenizer_path} to {repo_id}...")

    # Walk through all files in the directory
    for root, _, files in os.walk(local_tokenizer_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, local_tokenizer_path)

            # Upload the file
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=rel_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {rel_path}",
            )
            print(f"Uploaded {rel_path}")

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
