import datetime
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard
from transformers import DistilBertTokenizer

from src.model import DistilBertClassifier, save_model
from src.utils import encode_labels, encode_texts, load_training_conf

logger = logging.getLogger(__name__)


def training_data(
    tickets_data_path: str,
    text_column: str,
    label_column: str,
    test_size: float = 0.25,
    subset_size: int = -1,
    max_length: int = 100,
    pad_to_max_length: bool = True,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], DistilBertTokenizer, dict[int, str]]:
    df = pd.read_csv(tickets_data_path)
    x = df[text_column].tolist()
    y = df[label_column].tolist()
    unique_labels = sorted(list(set(y)))
    label_mapping = {i: str(label) for i, label in enumerate(unique_labels)}
    y = encode_labels(y, unique_labels)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
    tokenizer.max_length = max_length
    tokenizer.pad_to_max_length = pad_to_max_length
    logger.info("Tokenizing %d texts...", len(x))
    x = encode_texts(tokenizer, x)
    subset_size = len(x) if subset_size < 0 else subset_size
    x_train, x_test, y_train, y_test = train_test_split(
        x[:subset_size], y[:subset_size], test_size=test_size, random_state=42
    )
    return (x_train, x_test, y_train, y_test), tokenizer, label_mapping


def define_callbacks(
    patience: int = 3, min_delta: float = 0.01
) -> list[Callback]:
    training_early_stopper = EarlyStopping(
        monitor="val_accuracy",
        min_delta=min_delta,
        patience=patience,
        verbose=0,
        restore_best_weights=True,
    )
    training_tensorboard = TensorBoard(
        log_dir=os.path.join(
            "logs", "scalars", datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        )
    )
    return [training_early_stopper, training_tensorboard]


def train_model(
    model: DistilBertClassifier,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
    epochs: int = 1,
    callbacks: Optional[list[Callback]] = None,
) -> None:
    logger.info(
        "Starting training: epochs=%d, batch_size=%d, train_samples=%d",
        epochs, batch_size, len(x_train),
    )
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        max_queue_size=1000,
        workers=100,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
    )
    logger.info("Training complete.")


def run_training_pipeline(
    conf: dict,
    model_folder: str = ".",
    extra_callbacks: Optional[list[Callback]] = None,
) -> None:
    conf_train, conf_data = conf["training"], conf["data"]
    (x_train, x_test, y_train, y_test), tokenizer, label_mapping = training_data(
        conf_data["dataset_path"],
        conf_data["text_column"],
        conf_data["label_column"],
        test_size=conf_train.get("test_set_size", 0.25),
        subset_size=-1,
        max_length=conf_data["max_words_per_message"],
        pad_to_max_length=conf_data.get("pad_to_max_length", True),
    )
    model = DistilBertClassifier(
        num_labels=y_train.shape[1],
        learning_rate=conf_train.get("learning_rate", 5e-5),
    )
    callbacks = define_callbacks(
        patience=conf_train.get("early_stopping_patience", 3),
        min_delta=conf_train.get("early_stopping_min_delta_acc", 0.01),
    )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    train_model(
        model,
        x_train,
        x_test,
        y_train,
        y_test,
        epochs=conf_train.get("epochs", 1),
        batch_size=conf_train.get("batch_size", 64),
        callbacks=callbacks,
    )
    save_model(model, tokenizer, model_folder, label_mapping=label_mapping)
    logger.info("Model artifacts saved to %s", model_folder)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    conf = load_training_conf()
    run_training_pipeline(conf)
