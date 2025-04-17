import argparse
import os
import json

from huggingface_hub import HfApi
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
        # Convert tokens to proper format
        special_tokens_dict = {}
        for key, value in custom_tokens.items():
            if key == "additional_special_tokens":
                # For lists of additional special tokens
                if isinstance(value, list):
                    special_tokens_dict[key] = value
                else:
                    print(
                        f"Warning: 'additional_special_tokens' should be a list, got {type(value)}"
                    )
                    continue
            else:
                # For individual special tokens like bos_token, eos_token, etc.
                if isinstance(value, dict):
                    # If it's a dictionary of parameters, create an AddedToken
                    special_tokens_dict[key] = AddedToken(**value)
                else:
                    # If it's just a string
                    special_tokens_dict[key] = value

        # Add special tokens
        num_added = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added} custom tokens: {special_tokens_dict}")

        # Verify tokens were added
        print("Verifying special tokens:")
        for key, value in special_tokens_dict.items():
            if key == "additional_special_tokens":
                print(f"  {key}: {tokenizer.additional_special_tokens}")
            elif hasattr(tokenizer, key):
                print(f"  {key}: {getattr(tokenizer, key)}")

    # Set model max length
    tokenizer.model_max_length = max_length
    print(f"Set model max length to: {max_length}")

    # Set chat template
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    print("Set default chat template")

    # Create repository ID
    repo_id = f"{repository_owner}/{repository_name}"

    # Save the tokenizer locally first to ensure all changes are serialized
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tokenizer.save_pretrained(tmp_dir)
        print(
            f"Saved tokenizer temporarily to {tmp_dir} to ensure changes are serialized"
        )

        # Print the tokenizer_config.json contents
        tokenizer_config_path = os.path.join(tmp_dir, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, "r") as f:
                tokenizer_config = json.load(f)
            print("tokenizer_config.json content:")
            print(json.dumps(tokenizer_config, indent=2))
        else:
            print("tokenizer_config.json not found in the temporary directory")

        # Now push from the temporary directory
        tokenizer.push_to_hub(
            repo_id=repo_id,
            private=True,
            commit_message=f"{commit_message}",
        )

    # Upload all files from the tokenizer directory
    api = HfApi()
    print(f"Uploading all remaining files from {local_tokenizer_path} to {repo_id}...")

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
