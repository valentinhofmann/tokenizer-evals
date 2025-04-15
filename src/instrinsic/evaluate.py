import argparse
import datetime
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import regex as re
import uniseg.wordbreak
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

from src import utils


@dataclass
class TokenizerEvalResults:
    vocab_coverage: float
    compression_ratio: float
    avg_tokens_per_word: float
    avg_tokens_per_char: float
    oov_rate: float
    subword_stats: Dict[str, float]
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

        tokenized = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            tokenized.append(tokens)

        results = TokenizerEvalResults(
            vocab_coverage=self._calculate_vocab_coverage(tokenized),
            compression_ratio=self._calculate_compression_ratio(texts, tokenized),
            avg_tokens_per_word=self._calculate_tokens_per_word(texts, tokenized),
            avg_tokens_per_char=self._calculate_tokens_per_char(texts, tokenized),
            oov_rate=self._calculate_oov_rate(tokenized),
            subword_stats=self._analyze_subword_patterns(tokenized),
            token_distribution_stats=self._analyze_token_distribution(texts, tokenized),
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
        """Calculate compression ratio (bytes/tokens)"""
        total_bytes = sum(len(text.encode("utf-8")) for text in raw_texts)
        total_tokens = sum(len(tokens) for tokens in tokenized_texts)

        return total_bytes / total_tokens if total_tokens > 0 else 0

    def _calculate_tokens_per_word(
        self,
        raw_texts: List[str],
        tokenized_texts: List[List[int]],
        use_uniseg: bool = True,
    ) -> float:
        """Calculate average tokens per word"""
        if use_uniseg:
            total_words = sum(
                len(list(uniseg.wordbreak.words(text))) for text in raw_texts
            )
        else:
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
        unk_token_id = None

        if (
            hasattr(self.tokenizer, "unk_token")
            and self.tokenizer.unk_token is not None
        ):
            unk_token_id = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.unk_token
            )

        if unk_token_id is None:
            for common_unk in [
                "[UNK]",
                "<unk>",
                "UNK",
                "<unknown>",
                "UNKNOWN",
                "<oov>",
                "OOV",
            ]:
                try:
                    unk_id = self.tokenizer.convert_tokens_to_ids(common_unk)
                    # Make sure it's not just converting to a regular token
                    if (
                        unk_id is not None and unk_id != 0
                    ):  # Most tokenizers reserve 0 or None for padding
                        unk_token_id = unk_id
                        break
                except:
                    continue

        if unk_token_id is None:
            return 0.0

        total_tokens = sum(len(tokens) for tokens in tokenized_texts)
        unk_count = sum(tokens.count(unk_token_id) for tokens in tokenized_texts)
        return unk_count / total_tokens if total_tokens > 0 else 0

    def _analyze_subword_patterns(
        self, tokenized_texts: List[List[int]]
    ) -> Dict[str, float]:
        """Analyze subword tokenization patterns"""
        # Get the original token strings from token IDs
        token_strings = []
        for seq in tokenized_texts:
            for token_id in seq:
                try:
                    token_strings.append(self.tokenizer.convert_ids_to_tokens(token_id))
                except (AttributeError, TypeError):
                    token_strings.append(self.tokenizer.decode([token_id]))

        subword_counts = Counter()
        total_tokens = len(token_strings)

        bert_prefix = "##"
        sentencepiece_prefix = "▁"  # Often at the beginning of words
        byte_level_prefix = "Ġ"  # Used in GPT-2, RoBERTa

        for token in token_strings:
            if token.startswith(bert_prefix):
                subword_counts["subword"] += 1
            elif (
                sentencepiece_prefix in self.tokenizer.all_special_tokens
                or byte_level_prefix in self.tokenizer.all_special_tokens
            ):
                if (
                    not (
                        token.startswith(sentencepiece_prefix)
                        or token.startswith(byte_level_prefix)
                    )
                    and token not in self.tokenizer.all_special_tokens
                ):
                    subword_counts["subword"] += 1
                else:
                    subword_counts["full_word"] += 1

            # Fallback: if token has no spaces and isn't a special token, it's probably a subword
            elif (
                not re.search(r"\s", token)
                and len(token) > 0
                and token not in self.tokenizer.all_special_tokens
            ):
                # If it's a very short token (1-2 chars) or contains unusual characters, likely a subword
                if len(token) <= 2 or not token[0].isalnum():
                    subword_counts["subword"] += 1
                else:
                    subword_counts["full_word"] += 1
            else:
                subword_counts["full_word"] += 1

        return {
            "subword_ratio": (
                subword_counts["subword"] / total_tokens if total_tokens > 0 else 0
            ),
            "full_word_ratio": (
                subword_counts["full_word"] / total_tokens if total_tokens > 0 else 0
            ),
        }

    def _analyze_token_distribution(
        self, raw_texts: List[str], tokenized_texts: List[List[int]]
    ) -> Dict[str, float]:
        """Analyze token distribution statistics"""
        token_counts = Counter()

        for seq in tokenized_texts:
            token_counts.update(seq)

        for token_id in range(self.vocab_size):
            if token_id not in token_counts:
                token_counts[token_id] = 0

        total_tokens = sum(token_counts.values())
        token_probs = np.array(
            [count / total_tokens for count in token_counts.values()]
        )

        # For shannon we filter out zero probabilities to avoid log2(0)
        nonzero_probs = token_probs[token_probs > 0]
        shannon_entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
        alpha = 2.5
        renyi_entropy = (
            1 / (1 - alpha) * np.log2(np.sum(np.array(token_probs) ** alpha))
        )
        # Calculate word frequencies for uniseg word breakdown
        word_counts = Counter()
        for text in raw_texts:
            word_counts.update(list(uniseg.wordbreak.words(text)))

        # Calculate probability distribution
        total_words = sum(word_counts.values())
        word_probs = np.array([count / total_words for count in word_counts.values()])

        # Calculate Renyi entropy with the same alpha as token distribution
        uniseg_entropy = (
            1 / (1 - alpha) * np.log2(np.sum(word_probs**alpha))
            if total_words > 0
            else 0
        )

        return {
            "unique_token_ratio": len(token_counts) / total_tokens,
            "tokens.shannon.entropy": shannon_entropy,
            "tokens.renyi.entropy": renyi_entropy,
            "uniseg.renyi.entropy": uniseg_entropy,
            "top_10_token_ratio": sum(
                count for _, count in token_counts.most_common(10)
            )
            / total_tokens,
        }


def main():
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        "--name",
        type=str,
        default=None,
        help="Name of the tokenizer (default: tokenizer name from path)",
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

        console = Console(width=80, force_terminal=True)

        table = Table(
            title=f"Tokenizer Evaluation Results for {dataset_name}", expand=True
        )
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

        table.add_row("[bold]Token Distribution Statistics[/]", "")
        for k, v in results.token_distribution_stats.items():
            if ".entropy" in k:
                table.add_row(f"  {k}", f"{v:.2f}")
            else:
                table.add_row(f"  {k}", f"{v:.2%}")

        print()
        console.print(table)

    if args.write_file:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            tokenizer_name = tokenizer.name_or_path
        except:
            tokenizer_name = args.tokenizer

        safe_tokenizer_name = re.sub(r"[^\w\-_.]", "_", args.name or tokenizer_name)
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
                "distribution_stats": result.token_distribution_stats,
            }

        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
