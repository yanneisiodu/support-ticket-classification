import logging
import os
from typing import Any, Sequence, Tuple, Union

import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

from src.utils import encode_texts, load_label_mapping, save_label_mapping

logger = logging.getLogger(__name__)


class DistilBertClassifier(tf.keras.Model):
    def __init__(
        self,
        num_labels: int,
        learning_rate: float = 5e-5,
        dropout_rate: float = 0.2,
        metrics: list[str] | None = None,
    ):
        super().__init__()
        if metrics is None:
            metrics = ["accuracy"]
        hugging_face_distil_classifier = (
            TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-cased")
        )
        distil_classifier_out_dim = hugging_face_distil_classifier.config.dim
        self.distilbert = hugging_face_distil_classifier.get_layer("distilbert")

        self.dense1 = tf.keras.layers.Dense(
            distil_classifier_out_dim, activation="relu", name="dense1"
        )
        self.dense2 = tf.keras.layers.Dense(
            distil_classifier_out_dim // 2, activation="relu", name="dense2"
        )
        self.dense3 = tf.keras.layers.Dense(
            distil_classifier_out_dim // 4, activation="relu", name="dense3"
        )
        self.dense4 = tf.keras.layers.Dense(num_labels, name="dense4")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.compile(
            loss=loss_fn,
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=metrics,
        )

    def call(
        self, inputs: tf.Tensor, **kwargs: Any
    ) -> Tuple[tf.Tensor, Union[tf.Tensor, None]]:
        distilbert_output = self.distilbert(inputs, **kwargs)

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.dense1(pooled_output)
        pooled_output = self.dense2(pooled_output)
        pooled_output = self.dense3(pooled_output)
        logits = self.dense4(pooled_output)  # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]
        return outputs


def save_model(
    model: DistilBertClassifier,
    tokenizer: DistilBertTokenizer,
    model_folder: str = ".",
    label_mapping: dict[int, str] | None = None,
) -> None:
    os.makedirs(model_folder, exist_ok=True)
    model.save(os.path.join(model_folder, "my_model"))
    tokenizer.save_pretrained(os.path.join(model_folder, "tokenizer"))
    logger.info("Tokenizer saved via save_pretrained to %s/tokenizer", model_folder)
    if label_mapping is not None:
        save_label_mapping(label_mapping, os.path.join(model_folder, "label_mapping.json"))


def load_model(
    model_folder: str = ".",
) -> Tuple[DistilBertClassifier, DistilBertTokenizer, dict[int, str] | None]:
    model = tf.keras.models.load_model(os.path.join(model_folder, "my_model"))
    tokenizer_dir = os.path.join(model_folder, "tokenizer")
    if os.path.isdir(tokenizer_dir):
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_dir)
        logger.info("Tokenizer loaded via from_pretrained from %s", tokenizer_dir)
    else:
        import pickle

        pkl_path = os.path.join(model_folder, "tokenizer.pkl")
        logger.warning(
            "Loading tokenizer from legacy pickle at %s — re-save to migrate", pkl_path
        )
        with open(pkl_path, "rb") as f:
            tokenizer = pickle.load(f)
    label_mapping = None
    lm_path = os.path.join(model_folder, "label_mapping.json")
    if os.path.isfile(lm_path):
        label_mapping = load_label_mapping(lm_path)
        logger.info("Label mapping loaded from %s", lm_path)
    return model, tokenizer, label_mapping


def model_predict(
    model: DistilBertClassifier, tokenizer: DistilBertTokenizer, texts: Sequence[str]
):
    return model.predict(encode_texts(tokenizer, texts)).argmax(axis=1)
