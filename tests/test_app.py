from pathlib import Path

import pandas as pd
import pytest

import db
import ml

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()


@pytest.mark.usefixtures("database")
def test_a_new_database_has_no_history() -> None:
    assert db.load_history().empty


@pytest.mark.usefixtures("database")
def test_predictions_are_saved_with_every_field() -> None:
    db.save_prediction(900.0, 2, "Apartment", "Andheri", "Ready", "New", 12_500_000.0)

    row = db.load_history().iloc[0]

    assert row["region"] == "Andheri"
    assert row["predicted_price"] == 12_500_000.0


@pytest.mark.usefixtures("database")
def test_the_most_recent_prediction_comes_first() -> None:
    db.save_prediction(1.0, 1, "a", "first", "s", "n", 1.0)
    db.save_prediction(2.0, 2, "a", "second", "s", "n", 2.0)

    assert db.load_history()["region"].tolist() == ["second", "first"]


@pytest.mark.usefixtures("database")
def test_init_db_keeps_existing_rows() -> None:
    db.save_prediction(1.0, 1, "a", "kept", "s", "n", 1.0)

    db.init_db()

    assert len(db.load_history()) == 1


@pytest.fixture(scope="module")
def trained() -> tuple:
    previous = Path.cwd()
    import os

    os.chdir(ROOT)
    try:
        return ml.load_and_train()
    finally:
        os.chdir(previous)


def test_outliers_are_removed_from_the_training_data(trained: tuple) -> None:
    frame = trained[2]

    assert frame["area"].max() < 5000
    assert frame["bhk"].max() < 10


def test_rare_regions_are_grouped_as_other(trained: tuple) -> None:
    assert trained[2]["region_clean"].nunique() <= 51


def test_a_larger_home_is_predicted_to_cost_more(trained: tuple) -> None:
    model = trained[0]
    home = {
        "region_clean": "Other",
        "type": "Apartment",
        "status": "Ready to move",
        "age": "New",
    }

    small = model.predict(pd.DataFrame([{"bhk": 2, "area": 600, **home}]))[0]
    large = model.predict(pd.DataFrame([{"bhk": 4, "area": 3000, **home}]))[0]

    assert large > small
