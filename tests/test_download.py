import pandas as pd
import pytest

from download_data import SUBSET_DATA_URL, ALL_DATA_URL


def test_data_urls_are_defined():
    assert SUBSET_DATA_URL.startswith("https://")
    assert ALL_DATA_URL.startswith("https://")


def test_subset_url_points_to_csv():
    assert SUBSET_DATA_URL.endswith(".csv")


def test_all_url_points_to_csv():
    assert ALL_DATA_URL.endswith(".csv")
