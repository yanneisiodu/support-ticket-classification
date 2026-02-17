import json
import os
import tempfile

import numpy as np
import pytest

from src.utils import encode_labels, load_label_mapping, save_label_mapping


class TestEncodeLabels:
    def test_string_labels(self):
        labels = ["cat", "dog", "cat", "bird"]
        unique = ["bird", "cat", "dog"]
        result = encode_labels(labels, unique)
        assert result.shape == (4, 3)
        # "cat" -> index 1
        assert result[0].argmax() == 1
        # "dog" -> index 2
        assert result[1].argmax() == 2
        # "bird" -> index 0
        assert result[3].argmax() == 0

    def test_integer_labels(self):
        labels = [0, 1, 2, 0]
        unique = [0, 1, 2]
        result = encode_labels(labels, unique)
        assert result.shape == (4, 3)
        assert result[0].argmax() == 0
        assert result[1].argmax() == 1

    def test_single_label(self):
        labels = ["only"]
        unique = ["only"]
        result = encode_labels(labels, unique)
        assert result.shape == (1, 1)
        assert result[0][0] == 1.0

    def test_labels_sorted(self):
        labels = ["z", "a", "m"]
        unique = ["z", "a", "m"]
        result = encode_labels(labels, unique)
        # sorted unique: ["a", "m", "z"] -> a=0, m=1, z=2
        assert result[0].argmax() == 2  # "z"
        assert result[1].argmax() == 0  # "a"
        assert result[2].argmax() == 1  # "m"


class TestLabelMapping:
    def test_save_and_load(self, tmp_path):
        mapping = {0: "hardware", 1: "network", 2: "software"}
        path = str(tmp_path / "label_mapping.json")
        save_label_mapping(mapping, path)
        loaded = load_label_mapping(path)
        assert loaded == mapping

    def test_keys_are_ints(self, tmp_path):
        path = str(tmp_path / "label_mapping.json")
        with open(path, "w") as f:
            json.dump({"0": "a", "1": "b"}, f)
        loaded = load_label_mapping(path)
        assert all(isinstance(k, int) for k in loaded.keys())
