import argparse
import os
import json

from tokenizers import decoders
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
    tokenizer_class="gpt2",
):
    print(f"Loading tokenizer from: {local_tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        local_tokenizer_path, tokenizer_type=tokenizer_class
    )

    if custom_tokens and len(custom_tokens) > 0:
        # Convert tokens to proper format
        size_before_added = len(tokenizer.get_vocab())
        added_tokens_dict = {}
        for key, value in custom_tokens.items():
            if key == "additional_special_tokens":
                # For lists of additional special tokens
                if isinstance(value, list):
                    for token in value:
                        if isinstance(token, str):
                            # If it's a string, create an AddedToken
                            added_tokens_dict.setdefault(
                                "additional_special_tokens", []
                            ).append(AddedToken(token, lstrip=True, rstrip=True))
                        else:
                            print(
                                f"Warning: Invalid token type {type(token)} in 'additional_special_tokens'"
                            )
                    added_tokens_dict[key] = value
                else:
                    print(
                        f"Warning: 'additional_special_tokens' should be a list, got {type(value)}"
                    )
                    continue
            else:
                if isinstance(value, dict):
                    # If it's a dictionary of parameters, create an AddedToken
                    added_tokens_dict[key] = AddedToken(**value)
                else:
                    # If it's just a string
                    raise ValueError(
                        f"Invalid value for {key}: expected a dict for AddedToken, got {type(value)}"
                    )

        num_added = 0
        if "additional_special_tokens" in added_tokens_dict:
            # Add these as regular tokens
            tokens_to_add = added_tokens_dict.pop("additional_special_tokens")
            num_added += tokenizer.add_tokens(tokens_to_add)

        # Add any remaining tokens that were specified individually
        for key, token in added_tokens_dict.items():
            if isinstance(token, AddedToken):
                num_added += tokenizer.add_tokens(token)

        # You have to call this as well if you want them in your vocab
        tokenizer.add_special_tokens(added_tokens_dict)

        print(
            f"Added {num_added} tokens to the tokenizer (before: {size_before_added}, after: {len(tokenizer)})"
        )

    # Set model max length
    tokenizer.model_max_length = max_length
    print(f"Set model max length to: {max_length}")

    # Set chat template
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    print("Set default chat template")

    tokenizer.backend_tokenizer.decoder = decoders.ByteLevel(
        add_prefix_space=True,
        trim_offsets=True,
        use_regex=True,
    )
    print("Added ByteLevel decoder configuration!")

    # Create repository ID
    repo_id = f"{repository_owner}/{repository_name}"

    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tokenizer.save_pretrained(tmp_dir)
        print(
            f"Saved tokenizer temporarily to {tmp_dir} to ensure changes are serialized"
        )

        # Print the end of the vocab.json file to verify its content
        vocab_file_path = os.path.join(tmp_dir, "vocab.json")
        if os.path.exists(vocab_file_path):
            with open(vocab_file_path, "r") as f:
                vocab_json = json.load(f)
            print(f"\nVocabulary file contains {len(vocab_json)} tokens")
            # Get the last 10 items when sorted by token ID (value)
            sorted_items = sorted(vocab_json.items(), key=lambda x: x[1])[-10:]
            print("Last 10 entries in vocab.json:")
            for token, idx in sorted_items:
                print(f"  {idx}: {repr(token)}")
        else:
            print(f"No vocab.json found at {vocab_file_path}")

        print(f"Tokenizer vocabulary size: {len(tokenizer)}")
        print(f"Tokenizer model max length: {tokenizer.model_max_length}")
        print(f"Tokenizer chat template: {tokenizer.chat_template}")
        print(f"Tokenizer backend decoder: {tokenizer.backend_tokenizer.decoder}")
        print(
            f"Tokenizer additional special tokens: {tokenizer.additional_special_tokens}"
        )
        print(f"Tokenizer added tokens: {tokenizer.added_tokens_encoder}")
        print(f"Tokenizer repo ID: {repo_id}")
        print(f"Commit message: {commit_message}")

        # Print the last 10 tokens of the vocabulary
        vocab = tokenizer.get_vocab()
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        last_ten_tokens = sorted_vocab[-10:]
        print("\nLast 10 tokens in vocabulary:")
        for token, idx in last_ten_tokens:
            print(f"  {idx}: {repr(token)}")

        tokenizer.push_to_hub(
            repo_id=repo_id,
            private=True,
            commit_message=f"{commit_message}",
        )

    # # Upload all files from the tokenizer directory
    # api = HfApi()
    # print(f"Uploading all remaining files from {local_tokenizer_path} to {repo_id}...")

    # # Walk through all files in the directory
    # for root, _, files in os.walk(local_tokenizer_path):
    #     for file in files:
    #         file_path = os.path.join(root, file)
    #         rel_path = os.path.relpath(file_path, local_tokenizer_path)

    #         # Upload the file
    #         api.upload_file(
    #             path_or_fileobj=file_path,
    #             path_in_repo=rel_path,
    #             repo_id=repo_id,
    #             repo_type="model",
    #             commit_message=f"Upload {rel_path}",
    #         )
    #         print(f"Uploaded {rel_path}")

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
