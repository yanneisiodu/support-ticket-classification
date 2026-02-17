import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.utils import save_label_mapping


class TestSaveLoadModel:
    def test_save_creates_label_mapping(self, tmp_path):
        mapping = {0: "a", 1: "b", 2: "c"}
        lm_path = str(tmp_path / "label_mapping.json")
        save_label_mapping(mapping, lm_path)
        assert os.path.isfile(lm_path)
        with open(lm_path) as f:
            data = json.load(f)
        assert data == {"0": "a", "1": "b", "2": "c"}

    def test_load_model_legacy_fallback_warning(self, tmp_path, caplog):
        """Verify that loading from a pickle path emits a warning."""
        import pickle
        from unittest.mock import patch

        mock_tokenizer = MagicMock()
        pkl_path = tmp_path / "tokenizer.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(mock_tokenizer, f)

        model_dir = tmp_path / "my_model"
        model_dir.mkdir()

        with patch("src.model.tf.keras.models.load_model", return_value=MagicMock()):
            import logging
            with caplog.at_level(logging.WARNING):
                from src.model import load_model
                model, tokenizer, lm = load_model(str(tmp_path))
                assert "legacy pickle" in caplog.text
