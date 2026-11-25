import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def prepare_data(csv_path):
    """Charge, prépare et sépare les données."""
    df = pd.read_csv(csv_path)

    df = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Geography"])
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    return train_test_split(X, y, test_size=0.2, random_state=1)


def train_model(X_train, y_train):
    """Entraîne et retourne le modèle."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Calcule et retourne les performances du modèle."""
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }


def save_model(model, path):
    """Sauvegarde le modèle dans un fichier."""
    joblib.dump(model, path)


def load_model(path):
    """Charge et retourne un modèle sauvegardé."""
    return joblib.load(path)
