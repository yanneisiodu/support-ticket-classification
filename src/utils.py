import json
import logging
import os
from typing import Optional, Sequence, Union

import numpy as np
import yaml
from tensorflow.keras.utils import to_categorical
from transformers import DistilBertTokenizer

logger = logging.getLogger(__name__)


def load_training_conf(conf_path: Optional[str] = None) -> dict:
    conf_path = conf_path or os.path.join("src", "train_conf.yml")
    with open(conf_path, "r") as file:
        conf = yaml.full_load(file)
    return conf


def encode_texts(tokenizer: DistilBertTokenizer, texts: Sequence[str]) -> np.ndarray:
    max_length = getattr(tokenizer, "max_length", 512)
    pad_to_max = getattr(tokenizer, "pad_to_max_length", True)
    padding = "max_length" if pad_to_max else False
    return np.array(
        [
            tokenizer.encode(
                text,
                max_length=max_length,
                padding=padding,
                truncation=True,
            )
            for text in texts
        ]
    )


def encode_labels(
    texts_labels: Sequence[str], unique_labels: Sequence[Union[str, int]]
) -> np.ndarray:
    unique_labels = sorted(unique_labels)
    label_int = {label: i for i, label in enumerate(unique_labels)}
    if isinstance(unique_labels[0], str):
        texts_labels_encoded = np.array([label_int[label] for label in texts_labels])
    else:
        texts_labels_encoded = np.array(texts_labels)
    return to_categorical(texts_labels_encoded, num_classes=len(unique_labels))


def save_label_mapping(label_mapping: dict[int, str], path: str) -> None:
    with open(path, "w") as f:
        json.dump(label_mapping, f, indent=2)
    logger.info("Label mapping saved to %s", path)


def load_label_mapping(path: str) -> dict[int, str]:
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def pip_packages() -> list[str]:
    with open("requirements.txt") as f:
        pip_packages = "".join(f.readlines()).split(os.linesep)
    return [x for x in pip_packages if x != ""]
