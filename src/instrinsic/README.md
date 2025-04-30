This evaluation suite includes:

1. Basic tokenizer statistics:
   - Vocabulary coverage
   - Compression ratio
   - Tokens per word/character
   - OOV rate

2. Subword analysis:
   - Ratio of subwords to full words
   - Pattern analysis

3. Special case handling:
   - URLs
   - Email addresses
   - Numbers

4. Token distribution analysis:
   - Unique token ratio
   - Entropy
   - Top token concentration

To use this suite:

1. Initialize with your tokenizer:
```python
tokenizer = Tokenizer.from_file("path/to/tokenizer.json")
evaluator = TokenizerEvaluator(tokenizer)
```

2. Run evaluation on your texts:
```python
results = evaluator.evaluate_sample(your_texts)
```

3. Analyze results:
```python
print(f"Compression Ratio: {results.compression_ratio:.2f}")
print(f"OOV Rate: {results.oov_rate:.2%}")
```
