import argparse
import os
import shutil

from huggingface_hub import HfApi, login
from transformers import AutoTokenizer


def upload_tokenizer_to_hub(
    local_tokenizer_path, repository_owner, repository_name, tokenizer_name=None
):
    # Load the tokenizer from local directory
    print(f"Loading tokenizer from: {local_tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)

    # Check if merges.txt exists
    merges_path = os.path.join(local_tokenizer_path, "merges.txt")
    has_merges = os.path.exists(merges_path)
    if has_merges:
        print(f"Found merges.txt file at: {merges_path}")

    # Create repository ID
    repo_id = f"{repository_owner}/{repository_name}"

    # Push the tokenizer to the hub
    tokenizer.push_to_hub(
        repo_id=repo_id,
        private=True,
        commit_message=f"Upload tokenizer: {tokenizer_name or 'custom'}",
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
    parser.add_argument("--name", help="Tokenizer name (optional)")

    args = parser.parse_args()

    upload_tokenizer_to_hub(args.path, args.owner, args.repo, args.name)


if __name__ == "__main__":
    main()
