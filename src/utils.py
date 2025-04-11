import json

from datasets import load_dataset

SEED = 123
CHAT_DATASETS = {"wildchat", "chatbot_arena"}
CODE_DATASETS = {"mbpp", "human_eval"}
CORE_DATASETS = {"mmlu", "arc_challenge", "gsm8k"}


DATSET_TO_SPLIT = {
    "wildchat": "train",
    "chatbot_arena": "train",
    "mbpp": "test",
    "human_eval": "test",
    "mmlu": "auxiliary_train",
    "arc_challenge": "train",
    "gsm8k": "train",
}

DATSET_TO_SUBSET = {
    "mmlu": "all",
    "gsm8k": "main",
    "arc_challenge": "ARC-Challenge",
}

ALL_DATASETS = CHAT_DATASETS | CODE_DATASETS | CORE_DATASETS

FULL_NAMES = {
    "mbpp": "Muennighoff/mbpp",
    "human_eval": "openai/openai_humaneval",
    "wildchat": "allenai/WildChat",
    "chatbot_arena": "lmsys/chatbot_arena_conversations",
    "mmlu": "cais/mmlu",
    "arc_challenge": "allenai/ai2_arc",
    "gsm8k": "openai/gsm8k",
}

CODE_COLUMNS = {
    "mbpp": ["text", "code"],
    "human_eval": ["prompt", "canonical_solution"],
}

CORE_COLUMNS = {
    "mmlu": ["question", "choices"],
    "arc_challenge": ["question", "choices"],
    "gsm8k": ["question", "answer"],
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


def load_core_data(dataset_name, n_samples=1000):
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

    core_data_sample["text"] = core_data_sample.apply(
        lambda x: "\n".join(
            [
                "\n".join(x[col]) if col == "choices" else "\n".join(x[col])
                for col in CORE_COLUMNS[dataset_name]
            ]
        ),
        axis=1,
    )

    core_data_sample = core_data_sample[core_data_sample["text"].str.strip() != ""]

    return core_data_sample.sample(min(n_samples, len(core_data)), random_state=SEED)


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
        lambda x: "\n".join([x[col] for col in CODE_COLUMNS[dataset_name]]),
        axis=1,
    )

    code_data_sample = code_data_sample[code_data_sample["text"].str.strip() != ""]

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


def write_json(dataset_name, data, metric, tokenizer, output_dir=".output"):
    import os
    import re

    os.makedirs(output_dir, exist_ok=True)

    if metric not in data.columns:
        raise ValueError(f"Metric '{metric}' not found in dataset.")

    values = data[metric]
    out = {}
    safe_tokenizer_name = re.sub(r"[^\w\-_.]", "_", tokenizer.name_or_path)

    with open(
        f"{output_dir}/{safe_tokenizer_name}_{dataset_name}_{metric}.json", "w"
    ) as f:
        out["mean"] = values.mean()
        out["std"] = values.std()
        out["min"] = values.min()
        out["max"] = values.max()
        out["25pct"] = values.quantile(0.25)
        out["median"] = values.median()
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
