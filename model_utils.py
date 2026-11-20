
import os
import re
import json

import numpy as np
import pandas as pd
import torch
import streamlit as st
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# ============================================================
# 1. PROJECT PATHS AND DEVICE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_output")

MODEL_PATH = os.path.join(
    MODEL_DIR, "bilstm_toxic_comment_model.pt"
)

VOCAB_PATH = os.path.join(
    MODEL_DIR, "vocabulary.json"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR, "model_config.json"
)

METRICS_PATH = os.path.join(
    MODEL_DIR, "test_metrics.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. TEXT PREPROCESSING
# Exact logic from the Google Colab notebook
# ============================================================

def clean_text(text):
    """Normalize text using the original notebook pipeline."""

    text = str(text).lower()

    text = re.sub(
        r'https?://\S+|www\.\S+',
        ' urltoken ',
        text
    )

    text = re.sub(
        r'\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b',
        ' emailtoken ',
        text
    )

    text = re.sub(
        r"[^a-z0-9'\s]",
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text


# ============================================================
# 3. BiLSTM MODEL ARCHITECTURE
# Same architecture used during Colab training
# ============================================================

class BiLSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        dropout=0.35
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_dim * 2,
            output_dim
        )

    def forward(self, token_ids, lengths):

        embedded = self.dropout(
            self.embedding(token_ids)
        )

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        final_hidden = torch.cat(
            (hidden[-2], hidden[-1]),
            dim=1
        )

        return self.classifier(
            self.dropout(final_hidden)
        )


# ============================================================
# 4. LOAD MODEL AND SAVED ARTIFACTS
# ============================================================

@st.cache_resource
def load_model():

    required_files = [
        MODEL_PATH,
        VOCAB_PATH,
        CONFIG_PATH
    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Required model file not found: {path}"
            )

    # Load saved checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # Load vocabulary
    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        vocabulary = json.load(file)

    # Load model configuration
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(file)

    labels = checkpoint["label_columns"]

    thresholds = np.asarray(
        checkpoint["thresholds"],
        dtype=np.float32
    )

    max_length = int(
        checkpoint["max_length"]
    )

    # Validate saved artifacts
    if len(labels) != len(thresholds):
        raise ValueError(
            "Number of labels and thresholds do not match."
        )

    if len(labels) != checkpoint["output_dim"]:
        raise ValueError(
            "Saved output dimension does not match labels."
        )

    if len(vocabulary) != checkpoint["vocab_size"]:
        raise ValueError(
            "Vocabulary size does not match checkpoint."
        )

    if labels != config["label_columns"]:
        raise ValueError(
            "Label order differs between checkpoint and config."
        )

    if not np.allclose(
        thresholds,
        np.asarray(config["thresholds"], dtype=np.float32)
    ):
        raise ValueError(
            "Thresholds differ between checkpoint and config."
        )

    # Reconstruct the original BiLSTM architecture
    model = BiLSTMClassifier(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=checkpoint["output_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"]
    ).to(DEVICE)

    # Load trained weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return {
        "model": model,
        "vocabulary": vocabulary,
        "labels": labels,
        "thresholds": thresholds,
        "max_length": max_length,
        "config": config
    }


# ============================================================
# 5. CONVERT COMMENT INTO MODEL INPUT
# ============================================================

def encode_comment(comment, vocabulary, max_length):

    if not isinstance(comment, str):
        raise TypeError("Comment must be a string.")

    cleaned = clean_text(comment)

    tokens = cleaned.split()[:max_length]

    token_ids = [
        vocabulary.get(token, 1)
        for token in tokens
    ]

    # Handle empty comments
    if not token_ids:
        token_ids = [1]

    length = len(token_ids)

    # Padding
    token_ids += [0] * (max_length - length)

    token_tensor = torch.tensor(
        [token_ids],
        dtype=torch.long
    )

    length_tensor = torch.tensor(
        [length],
        dtype=torch.long
    )

    return token_tensor, length_tensor


# ============================================================
# 6. PREDICT TOXICITY
# ============================================================

def predict_toxicity(comment, artifacts):

    if not isinstance(comment, str) or not comment.strip():
        raise ValueError("Please enter a non-empty comment.")

    model = artifacts["model"]
    vocabulary = artifacts["vocabulary"]
    labels = artifacts["labels"]
    thresholds = artifacts["thresholds"]
    max_length = artifacts["max_length"]

    token_ids, lengths = encode_comment(
        comment,
        vocabulary,
        max_length
    )

    token_ids = token_ids.to(DEVICE)
    lengths = lengths.to(DEVICE)

    with torch.inference_mode():

        logits = model(
            token_ids,
            lengths
        )

        probabilities = torch.sigmoid(
            logits
        )[0].cpu().numpy()

    predictions = probabilities >= thresholds

    results = pd.DataFrame({

        "Toxicity Category": labels,

        "Probability": probabilities,

        "Threshold": thresholds,

        "Prediction": np.where(
            predictions,
            "Detected",
            "Not Detected"
        )

    })

    results = results.sort_values(
        "Probability",
        ascending=False
    ).reset_index(drop=True)

    overall_result = (
        "Potentially Toxic"
        if predictions.any()
        else "No Toxicity Label Detected"
    )

    return overall_result, results
