
from pathlib import Path
from model_pipeline import prepare_data


def test_prepare_data():
    csv_path = Path(__file__).parent / "Churn_Modelling.csv"

    X_train, X_test, y_train, y_test = prepare_data(str(csv_path))

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
