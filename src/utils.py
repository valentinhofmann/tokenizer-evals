import json
import logging

from datasets import load_dataset

logger = logging.getLogger(__name__)

SEED = 123
CHAT_DATASETS = {"wildchat", "chatbot_arena"}
CODE_DATASETS = {"mbpp", "human_eval"}
CORE_DATASETS = {"mmlu", "arc_challenge", "gsm8k", "flores200"}

DATSET_TO_SPLIT = {
    "arc_challenge": "train",
    "chatbot_arena": "train",
    "flores200": "dev",
    "gsm8k": "train",
    "human_eval": "test",
    "mbpp": "test",
    "mmlu": "auxiliary_train",
    "wildchat": "train",
}

DATSET_TO_SUBSET = {
    "arc_challenge": "ARC-Challenge",
    "gsm8k": "main",
    "mmlu": "all",
    "flores200": "all",
}

ALL_DATASETS = CHAT_DATASETS | CODE_DATASETS | CORE_DATASETS

FULL_NAMES = {
    "arc_challenge": "allenai/ai2_arc",
    "chatbot_arena": "lmsys/chatbot_arena_conversations",
    "flores200": "facebook/flores",
    "gsm8k": "openai/gsm8k",
    "human_eval": "openai/openai_humaneval",
    "mbpp": "Muennighoff/mbpp",
    "mmlu": "cais/mmlu",
    "wildchat": "allenai/WildChat",
}

CODE_COLUMNS = {
    "mbpp": ["text", "code"],
    "human_eval": ["prompt", "canonical_solution"],
}

CORE_COLUMNS = {
    "mmlu": ["question"],
    "arc_challenge": ["question"],
    "gsm8k": ["question"],
}

CUSTOM_SELECT = {
    "non-english": {
        "flores200": lambda df: df.assign(
            text=df.filter(regex="^sentence_")
            .sample(n=min(5, len(df.filter(regex="^sentence_").columns)), axis=1)
            .apply(lambda x: "\n".join(x), axis=1)
        )
    },
    "english-only": {},
}

CONVERSATION_COLS = {"wildchat": "conversation", "chatbot_arena": "conversation_a"}


def load_data(dataset_name, n_samples=1000, english_only=True):
    if dataset_name in CHAT_DATASETS:
        return load_chat_data(dataset_name, n_samples, english_only)
    elif dataset_name in CODE_DATASETS:
        return load_code_data(dataset_name, n_samples)
    elif dataset_name in CORE_DATASETS:
        return load_core_data(dataset_name, n_samples)
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not supported yet.")


def load_core_data(dataset_name, n_samples=1000, english_only=False):
    core_data = load_dataset(
        FULL_NAMES[dataset_name],
        DATSET_TO_SUBSET[dataset_name],
        split=DATSET_TO_SPLIT[dataset_name],
    )

    core_data_sample = (
        core_data.shuffle(seed=SEED)
        .select(range(min(n_samples * 3, len(core_data))))
        .to_pandas()
    )

    logger.info(
        f"Loaded {len(core_data_sample)} samples from {dataset_name} dataset..."
    )

    custom_select = CUSTOM_SELECT.get(
        "non-english" if not english_only else "english-only", {}
    )

    if dataset_name in custom_select:
        # Apply custom selection logic for specific datasets,
        # but because these datasets can be large we
        # sample first and then apply the custom selection
        # to avoid loading the entire dataset into memory.
        core_data_sample = custom_select[dataset_name](
            core_data_sample.sample(
                min(n_samples, len(core_data_sample)), random_state=SEED
            )
        )

    else:
        core_data_sample["text"] = core_data_sample.apply(
            lambda x: "\n".join(
                [
                    " ".join(x[col]) if col == "choices" else "".join(x[col])
                    for col in CORE_COLUMNS[dataset_name]
                ]
            ),
            axis=1,
        )

    core_data_sample = core_data_sample[core_data_sample["text"].str.strip() != ""]
    print_examples(core_data_sample, dataset_name)

    return core_data_sample.sample(
        min(n_samples, len(core_data_sample)), random_state=SEED
    )


def load_code_data(dataset_name, n_samples=1000):
    code_data = load_dataset(
        FULL_NAMES[dataset_name], split=DATSET_TO_SPLIT[dataset_name]
    )

    code_data_sample = (
        code_data.shuffle(seed=SEED)
        .select(range(min(n_samples * 3, len(code_data))))
        .to_pandas()
    )

    code_data_sample["text"] = code_data_sample.apply(
        lambda x: "\n\n".join([x[col] for col in CODE_COLUMNS[dataset_name]]),
        axis=1,
    )

    code_data_sample = code_data_sample[code_data_sample["text"].str.strip() != ""]
    logger.info(
        f"Loaded {len(code_data_sample)} samples from {dataset_name} dataset..."
    )
    print_examples(code_data_sample, dataset_name)

    return code_data_sample.sample(min(n_samples, len(code_data)), random_state=SEED)


def load_chat_data(dataset_name, n_samples=1000, english_only=True):
    chat_data = load_dataset(
        FULL_NAMES[dataset_name], split=DATSET_TO_SPLIT[dataset_name]
    )

    # Pre-sample larger chunk
    chat_data_sample = (
        chat_data.shuffle(seed=SEED).select(range(n_samples * 3)).to_pandas()
    )

    # Process user input message
    chat_data_sample["text"] = chat_data_sample[CONVERSATION_COLS[dataset_name]].apply(
        lambda x: x[0]["content"]
    )
    chat_data_sample = chat_data_sample[
        chat_data_sample["text"].apply(lambda x: x.strip() != "")
    ]

    # Optional: filter non-English messages
    if english_only:
        chat_data_sample = chat_data_sample[
            chat_data_sample["language"].str.lower() == "english"
        ]

    # Final sampling
    logger.info(
        f"Loaded {len(chat_data_sample)} samples from {dataset_name} dataset..."
    )
    print_examples(chat_data_sample, dataset_name)
    return chat_data_sample.sample(n_samples, random_state=SEED)


def compute_metrics(data, tokenizer, text_col="text"):

    # Split into words and tokens
    data["tokens"] = data[text_col].apply(lambda x: tokenizer.tokenize(x))
    data["words"] = data[text_col].apply(lambda x: x.split())

    # Compute number of words and tokens
    data["n_tokens"] = data["tokens"].apply(len)
    data["n_words"] = data["words"].apply(len)

    # Compute fertility
    data["fertility"] = data.apply(
        lambda r: r["n_tokens"] / r["n_words"] if r["n_words"] > 0 else 0, axis=1
    )
    return data


def print_examples(data, dataset_name, n=5):
    logger.info(f"Example instances from {dataset_name}...")

    for text in data["text"].sample(n).to_list():
        logger.info(f"\n{text}\n")


def write_json(name, results, metric, tokenizer, output_dir=".output"):
    import os
    import re

    os.makedirs(output_dir, exist_ok=True)
    safe_tokenizer_name = re.sub(r"[^\w\-_.]", "_", tokenizer.name_or_path)

    out = {safe_tokenizer_name: {}}
    for key, data in results.items():

        if metric not in data.columns:
            raise ValueError(f"Metric '{metric}' not found in dataset.")

        to_update = out[safe_tokenizer_name].setdefault(key, {})

        values = data[metric]

        to_update["mean"] = values.mean()
        to_update["std"] = values.std()
        to_update["min"] = values.min()
        to_update["max"] = values.max()
        to_update["25pct"] = values.quantile(0.25)
        to_update["median"] = values.median()

    with open(f"{output_dir}/{safe_tokenizer_name}_{name}_{metric}.json", "w") as f:
        f.write(json.dumps(out, indent=4))


def display_metric(dataset_name, data, metric):
    if metric not in data.columns:
        raise ValueError(f"Metric '{metric}' not found in dataset.")
    values = data[metric]

    # Print metric statistics
    print(f"--- {metric.upper()} statistics on {dataset_name} ---")
    print(f"Mean:     {values.mean():.3f}")
    print(f"Std Dev:  {values.std():.3f}")
    print(f"Min:      {values.min():.3f}")
    print(f"Max:      {values.max():.3f}")
    print(f"25%:      {values.quantile(0.25):.3f}")
    print(f"Median:   {values.median():.3f}")
    print(f"75%:      {values.quantile(0.75):.3f}")
