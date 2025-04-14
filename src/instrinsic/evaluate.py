import argparse
import datetime
import json
import logging
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import regex as re
from transformers import AutoTokenizer

from src import utils
from rich.console import Console
from rich.table import Table


@dataclass
class TokenizerEvalResults:
    vocab_coverage: float
    compression_ratio: float
    avg_tokens_per_word: float
    avg_tokens_per_char: float
    oov_rate: float
    subword_stats: Dict[str, float]
    special_case_metrics: Dict[str, float]
    token_distribution_stats: Dict[str, float]


class TokenizerEvaluator:
    def __init__(self, tokenizer: AutoTokenizer):
        """Initialize the evaluator with a tokenizer"""
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.vocab = tokenizer.get_vocab()

    def evaluate_sample(
        self, texts: List[str], sample_size: int = 10000
    ) -> TokenizerEvalResults:
        """Run all evaluations on a sample of texts"""
        if len(texts) > sample_size:
            texts = random.sample(texts, sample_size)

        tokenized = [self.tokenizer.encode(text) for text in texts]

        results = TokenizerEvalResults(
            vocab_coverage=self._calculate_vocab_coverage(tokenized),
            compression_ratio=self._calculate_compression_ratio(texts, tokenized),
            avg_tokens_per_word=self._calculate_tokens_per_word(texts, tokenized),
            avg_tokens_per_char=self._calculate_tokens_per_char(texts, tokenized),
            oov_rate=self._calculate_oov_rate(tokenized),
            subword_stats=self._analyze_subword_patterns(tokenized),
            special_case_metrics=self._evaluate_special_cases(texts),
            token_distribution_stats=self._analyze_token_distribution(tokenized),
        )

        return results

    def _calculate_vocab_coverage(self, tokenized_texts: List[List[int]]) -> float:
        """Calculate what percentage of vocabulary tokens that are actually used"""
        used_tokens = set()
        for seq in tokenized_texts:
            used_tokens.update(seq)
        return len(used_tokens) / self.vocab_size

    def _calculate_compression_ratio(
        self, raw_texts: List[str], tokenized_texts: List[List[int]]
    ) -> float:
        """Calculate compression ratio (chars/tokens)"""
        total_chars = sum(len(text) for text in raw_texts)
        total_tokens = sum(len(tokens) for tokens in tokenized_texts)
        return total_chars / total_tokens if total_tokens > 0 else 0

    def _calculate_tokens_per_word(
        self, raw_texts: List[str], tokenized_texts: List[List[int]]
    ) -> float:
        """Calculate average tokens per word"""
        total_words = sum(len(text.split()) for text in raw_texts)
        total_tokens = sum(len(tokens) for tokens in tokenized_texts)
        return total_tokens / total_words if total_words > 0 else 0

    def _calculate_tokens_per_char(
        self, raw_texts: List[str], tokenized_texts: List[List[int]]
    ) -> float:
        """Calculate average tokens per character"""
        total_chars = sum(len(text) for text in raw_texts)
        total_tokens = sum(len(tokens) for tokens in tokenized_texts)
        return total_tokens / total_chars if total_chars > 0 else 0

    def _calculate_oov_rate(self, tokenized_texts: List[List[int]]) -> float:
        """Calculate OOV rate using unknown token ID"""
        unk_token_id = self.tokenizer.convert_tokens_to_ids("[UNK]")
        if unk_token_id is None:
            return 0.0

        total_tokens = sum(len(tokens) for tokens in tokenized_texts)
        unk_count = sum(tokens.count(unk_token_id) for tokens in tokenized_texts)
        return unk_count / total_tokens if total_tokens > 0 else 0

    def _analyze_subword_patterns(
        self, tokenized_texts: List[List[int]]
    ) -> Dict[str, float]:
        """Analyze subword tokenization patterns"""
        token_strings = []
        for seq in tokenized_texts:
            token_strings.extend(self.tokenizer.decode(seq).split())

        subword_counts = Counter()
        total_tokens = 0

        for token in token_strings:
            if token.startswith("##") or token.endswith("##"):
                subword_counts["subword"] += 1
            else:
                subword_counts["full_word"] += 1
            total_tokens += 1

        return {
            "subword_ratio": (
                subword_counts["subword"] / total_tokens if total_tokens > 0 else 0
            ),
            "full_word_ratio": (
                subword_counts["full_word"] / total_tokens if total_tokens > 0 else 0
            ),
        }

    def _evaluate_special_cases(self, texts: List[str]) -> Dict[str, float]:
        """Evaluate handling of special cases"""
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        email_pattern = r"[\w\.-]+@[\w\.-]+"
        number_pattern = r"\d+"

        special_cases = {"urls": 0, "emails": 0, "numbers": 0}

        for text in texts:
            special_cases["urls"] += len(re.findall(url_pattern, text))
            special_cases["emails"] += len(re.findall(email_pattern, text))
            special_cases["numbers"] += len(re.findall(number_pattern, text))

        total_texts = len(texts)
        return {k: v / total_texts for k, v in special_cases.items()}

    def _analyze_token_distribution(
        self, tokenized_texts: List[List[int]]
    ) -> Dict[str, float]:
        """Analyze token distribution statistics"""
        token_counts = Counter()
        for seq in tokenized_texts:
            token_counts.update(seq)

        total_tokens = sum(token_counts.values())
        probs = [count / total_tokens for count in token_counts.values()]

        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        return {
            "unique_token_ratio": len(token_counts) / self.vocab_size,
            "entropy": entropy,
            "top_10_token_ratio": sum(
                count for _, count in token_counts.most_common(10)
            )
            / total_tokens,
        }


def main():
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Evaluate intrinsic tokenizer properties (vocab coverage, compression ratio, etc.)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer.json or HuggingFace tokenizer name",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of samples to use (default: 1000)",
    )
    parser.add_argument(
        "--english_only",
        action="store_true",
        help="Use only English samples for chat datasets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to evaluate (default: evaluate all)",
    )
    parser.add_argument(
        "--write_file", action="store_true", help="Write results to a file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=".output", help="Output directory for results"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    evaluator = TokenizerEvaluator(tokenizer)

    datasets_to_process = [args.dataset] if args.dataset else sorted(utils.ALL_DATASETS)

    all_results = {}

    for dataset_name in datasets_to_process:
        logger.info(f"Evaluating tokenizer {args.tokenizer} on dataset: {dataset_name}")

        # Load dataset
        data = utils.load_data(dataset_name, args.n_samples, args.english_only)
        texts = data["text"].tolist()

        # Evaluate tokenizer on dataset
        results = evaluator.evaluate_sample(texts)
        all_results[dataset_name] = results

        console = Console()

        table = Table(title=f"Tokenizer Evaluation Results for {dataset_name}")
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Vocabulary Coverage", f"{results.vocab_coverage:.2%}")
        table.add_row("Compression Ratio", f"{results.compression_ratio:.2f}")
        table.add_row("Avg Tokens per Word", f"{results.avg_tokens_per_word:.2f}")
        table.add_row("Avg Tokens per Char", f"{results.avg_tokens_per_char:.2f}")
        table.add_row("OOV Rate", f"{results.oov_rate:.2%}")

        table.add_row(
            "[bold]Subword Statistics[/]", ""
        )  # Subheading - need to use markdown for bold
        for k, v in results.subword_stats.items():
            table.add_row(f"  {k}", f"{v:.2%}")

        table.add_row("[bold]Special Case Metrics[/]", "")
        for k, v in results.special_case_metrics.items():
            table.add_row(f"  {k}", f"{v:.2f}")

        table.add_row("[bold]Token Distribution Statistics[/]", "")
        for k, v in results.token_distribution_stats.items():
            table.add_row(f"  {k}", f"{v:.2f}")

        console.print(table)

    if args.write_file:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            tokenizer_name = tokenizer.name_or_path
        except:
            tokenizer_name = args.tokenizer

        safe_tokenizer_name = re.sub(r"[^\w\-_.]", "_", tokenizer_name)
        output_file = f"{args.output_dir}/{safe_tokenizer_name}_eval_{timestamp}.json"

        serializable_results = {}
        for dataset_name, result in all_results.items():
            serializable_results[dataset_name] = {
                "vocab_coverage": result.vocab_coverage,
                "compression_ratio": result.compression_ratio,
                "avg_tokens_per_word": result.avg_tokens_per_word,
                "avg_tokens_per_char": result.avg_tokens_per_char,
                "oov_rate": result.oov_rate,
                "subword_stats": result.subword_stats,
                "special_case_metrics": result.special_case_metrics,
                "token_distribution_stats": result.token_distribution_stats,
            }

        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
