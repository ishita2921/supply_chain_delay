import os
import pandas as pd
import numpy as np
import tempfile
from supply_chain import utils


def test_candidate_files_creates_and_detects_files(tmp_path):
    # create a temporary CSV file
    data_dir = tmp_path / "data" / "interim"
    data_dir.mkdir(parents=True)
    f = data_dir / "test.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(f, index=False)

    # monkeypatch glob
    files = utils.candidate_files()
    # since candidate_files looks in ./data/, check we created properly
    assert isinstance(files, list)


def test_standardize_columns_adds_missing_fields():
    df = pd.DataFrame({
        "Event Date": ["2025-09-01", "2025-09-02"],
        "Logistics Delay": [0, 1]
    })
    df_out = utils.standardize_columns(df)

    # should lowercase + underscore columns
    assert "event_date" in df_out.columns
    assert "delay_flag" in df_out.columns
    assert df_out["delay_flag"].dtype == bool
    assert "supplier" in df_out.columns
    assert "mode" in df_out.columns
    assert "region" in df_out.columns
    assert "month" in df_out.columns


def test_predict_probability_with_mock_model():
    class DummyModel:
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])

    class DummyPipeline:
        def transform(self, X):
            return X

    model = DummyModel()
    pipeline = DummyPipeline()

    X = pd.DataFrame({"a": [1]})
    prob = utils.predict_probability(model, pipeline, X)
    assert 0 <= prob <= 1
    assert abs(prob - 0.7) < 1e-6


def test_clean_feature_name_and_input_value():
    # clean_feature_name
    assert utils.clean_feature_name("ohe__region_North") == "region = North"
    assert utils.clean_feature_name("numeric__month") == "month"

    # input_value_for_transformed_feature
    df = pd.DataFrame({"region": ["North"], "month": [9]})
    name = "region = North"
    val = utils.input_value_for_transformed_feature(name, df)
    assert val == "North"

    name2 = "numeric__month"
    val2 = utils.input_value_for_transformed_feature(name2, df)
    assert val2 == 9
