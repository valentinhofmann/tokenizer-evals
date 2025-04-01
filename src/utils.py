from datasets import load_dataset

CHAT_DATASETS = {
    "wildchat", 
    "chatbot_arena"
}

FULL_NAMES = {
    "wildchat": "allenai/WildChat",
    "chatbot_arena": "lmsys/chatbot_arena_conversations"
}

CONVERSATION_COLS = {
    "wildchat": "conversation",
    "chatbot_arena": "conversation_a"
}


def load_data(dataset_name, n_samples=1000, english_only=True):
    if dataset_name in CHAT_DATASETS:
        return load_chat_data(dataset_name, n_samples, english_only)
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not supported yet.")


def load_chat_data(dataset_name, n_samples=1000, english_only=True):
    chat_data = load_dataset(FULL_NAMES[dataset_name], split="train")

    # Pre-sample larger chunk
    chat_data_sample = chat_data.shuffle(seed=123).select(range(n_samples * 3)).to_pandas()

    # Process user input message
    chat_data_sample["user_input"] = chat_data_sample[CONVERSATION_COLS[dataset_name]].apply(lambda x: x[0]["content"])
    chat_data_sample = chat_data_sample[chat_data_sample["user_input"].apply(lambda x: x.strip() != "")]

    # Optional: filter non-English messages
    if english_only:
        chat_data_sample = chat_data_sample[chat_data_sample["language"].str.lower() == "english"]

    # Final sampling
    return chat_data_sample.sample(n_samples, random_state=123)


def compute_metrics(data, tokenizer, text_col="user_input"):

    # Split into words and tokens
    data["tokens"] = data[text_col].apply(lambda x: tokenizer.tokenize(x))
    data["words"] = data[text_col].apply(lambda x: x.split())

    # Compute number of words and tokens
    data["n_tokens"] = data["tokens"].apply(len)
    data["n_words"] = data["words"].apply(len)

    # Compute fertility
    data["fertility"] = data.apply(
        lambda r: r["n_tokens"] / r["n_words"] if r["n_words"] > 0 else 0, 
        axis=1
    )
    return data


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
