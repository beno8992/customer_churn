import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, task

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# =====================================
# CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = str(BASE_DIR / "Churn_Modelling.csv")
MODEL_PATH = str(BASE_DIR / "classifier.joblib")

CODE_FILES = [
    str(BASE_DIR / "model_pipeline.py"),
    str(BASE_DIR / "main.py"),
    str(BASE_DIR / "pipeline_prefect.py"),
]


# =====================================
# TACHES DE QUALITE DU CODE
# =====================================


@task(name="Installation des dependances")
def install_task():
    print("Installation des dependances...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(BASE_DIR / "requirements.txt"),
        ],
        check=True,
        cwd=BASE_DIR,
    )


@task(name="Formatage du code")
def format_task():
    print("Formatage du code avec Black...")

    subprocess.run(
        [sys.executable, "-m", "black", *CODE_FILES],
        check=True,
        cwd=BASE_DIR,
    )


@task(name="Qualite du code")
def quality_task():
    print("Verification avec Flake8...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "flake8",
            "--max-line-length=88",
            "--extend-ignore=E203,W503",
            *CODE_FILES,
        ],
        check=True,
        cwd=BASE_DIR,
    )


@task(name="Securite du code")
def security_task():
    print("Analyse avec Bandit...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "bandit",
            "-q",
            "-ll",
            *CODE_FILES,
        ],
        check=True,
        cwd=BASE_DIR,
    )


@task(name="Tests unitaires")
def tests_task():
    print("Execution des tests unitaires...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(BASE_DIR / "test_model_pipeline.py"),
        ],
        check=True,
        cwd=BASE_DIR,
    )


# =====================================
# TACHES MACHINE LEARNING
# =====================================


@task(name="Preparation des donnees")
def prepare_task():
    print("Preparation des donnees...")

    return prepare_data(DATA_PATH)


@task(name="Entrainement du modele")
def train_task(X_train, y_train):
    print("Entrainement du modele...")

    return train_model(X_train, y_train)


@task(name="Sauvegarde du modele")
def save_task(model):
    print("Sauvegarde du modele...")

    save_model(model, MODEL_PATH)


@task(name="Chargement du modele")
def load_task():
    print("Chargement du modele...")

    return load_model(MODEL_PATH)


@task(name="Evaluation du modele")
def evaluate_task(model, X_test, y_test):
    print("Evaluation du modele...")

    return evaluate_model(model, X_test, y_test)


@task(name="Prediction")
def predict_task(model, X_test):
    print("Prediction...")

    predictions = model.predict(X_test[:5])

    print("Predictions :", predictions)

    return predictions


# =====================================
# FLOW CODE
# =====================================


@flow(name="code", log_prints=True)
def code_flow():
    print("Verification du code...")

    install_task()
    format_task()
    quality_task()
    security_task()
    tests_task()

    print("Verification du code terminee !")


# =====================================
# FLOW TRAIN
# =====================================


@flow(name="train", log_prints=True)
def train_flow():
    print("Debut de l'entrainement...")

    X_train, _, y_train, _ = prepare_task()

    train_task(X_train, y_train)

    print("Entrainement termine !")


# =====================================
# FLOW EVALUATE
# =====================================


@flow(name="evaluate", log_prints=True)
def evaluate_flow():
    print("Debut de l'evaluation...")

    _, X_test, _, y_test = prepare_task()

    model = load_task()

    evaluate_task(model, X_test, y_test)

    print("Evaluation terminee !")


# =====================================
# FLOW ALL
# =====================================


@flow(name="all", log_prints=True)
def all_flow():
    print("Debut du pipeline complet...")

    # Controle du code
    code_flow()

    # Preparation
    X_train, X_test, y_train, y_test = prepare_task()

    # Entrainement
    model = train_task(X_train, y_train)

    # Sauvegarde
    save_task(model)

    # Evaluation
    evaluate_task(model, X_test, y_test)

    # Prediction
    predict_task(model, X_test)

    print("Pipeline ALL termine !")


# =====================================
# PROGRAMME PRINCIPAL
# =====================================


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML avec Prefect")

    parser.add_argument(
        "--flow",
        choices=[
            "all",
            "train",
            "entrainement",
            "evaluate",
            "code",
        ],
        required=True,
        help="Choisir le flow a executer",
    )

    args = parser.parse_args()

    if args.flow == "all":
        all_flow()

    elif args.flow in ("train", "entrainement"):
        train_flow()

    elif args.flow == "evaluate":
        evaluate_flow()

    elif args.flow == "code":
        code_flow()


if __name__ == "__main__":
    main()
