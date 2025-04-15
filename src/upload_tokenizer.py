from transformers import AutoTokenizer
from huggingface_hub import login
import argparse


def upload_tokenizer_to_hub(
    local_tokenizer_path, repository_owner, repository_name, tokenizer_name=None
):
    login()

    # Load the tokenizer from local directory
    print(f"Loading tokenizer from: {local_tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)

    # Create repository ID
    repo_id = f"{repository_owner}/{repository_name}"

    # Push the tokenizer to the hub as a private repository
    tokenizer.push_to_hub(
        repo_id=repo_id,
        private=True,
        commit_message=f"Upload tokenizer: {tokenizer_name or 'custom'}",
    )

    print(f"Successfully uploaded tokenizer to: {repo_id} (private repository)")


if __name__ == "__main__":
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
