import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def display_results(model, X_test, y_test):
    """Affiche les performances du modèle."""
    results = evaluate_model(model, X_test, y_test)
    print(f"Accuracy : {results['accuracy']:.2%}")
    print("Matrice de confusion (classes : 0, 1) :")
    print(results["confusion_matrix"])


def main():
    parser = argparse.ArgumentParser(description="Pipeline de prédiction du churn")
    parser.add_argument(
        "--action",
        choices=["train", "evaluate", "all"],
        required=True,
    )
    parser.add_argument("--data", default="Churn_Modelling.csv")
    parser.add_argument("--model", default="classifier.joblib")
    args = parser.parse_args()

    print("\n=== PRÉPARATION DES DONNÉES ===")
    X_train, X_test, y_train, y_test = prepare_data(args.data)

    print(f"Clients pour l'entraînement : {len(X_train)}")
    print(f"Clients pour le test : {len(X_test)}")
    print(f"Variables utilisées : {list(X_train.columns)}")
    print("\nAperçu des données d'entraînement :")
    print(X_train.head())

    if args.action in ["train", "all"]:
        print("\n=== ENTRAÎNEMENT ===")
        model = train_model(X_train, y_train)
        print(model)
        print("Entraînement terminé.")

        print("\n=== ÉVALUATION ===")
        display_results(model, X_test, y_test)

        print("\n=== SAUVEGARDE ===")
        save_model(model, args.model)
        print(f"Modèle enregistré : {args.model}")

    if args.action in ["evaluate", "all"]:
        print("\n=== CHARGEMENT DU MODÈLE ===")
        loaded_model = load_model(args.model)
        print(f"Modèle chargé : {args.model}")

        print("\n=== ÉVALUATION DU MODÈLE CHARGÉ ===")
        display_results(loaded_model, X_test, y_test)

    print("\n=== EXÉCUTION TERMINÉE ===")


if __name__ == "__main__":
    main()
