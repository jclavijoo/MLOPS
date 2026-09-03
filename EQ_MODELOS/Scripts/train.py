# Modelos implementados para entrenamiento:
# Random Forest y Regresión Logística

from pathlib import Path
from abc import ABC, abstractmethod
import argparse
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "species"
DROP_COLUMNS = ["rowid"]


class BasePenguinModel(ABC):
    """
    Clase base para cada modelo.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipeline = None

    @property
    @abstractmethod
    def name(self):
        """Nombre utilizado para identificar y guardar el modelo."""
        pass

    @abstractmethod
    def build_estimator(self):
        """Devuelve el estimador de scikit-learn."""
        pass

    def build_preprocessor(self, X):
        numeric_features = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore")
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_pipeline,
                    numeric_features
                ),
                (
                    "cat",
                    categorical_pipeline,
                    categorical_features
                ),
            ]
        )

    def build_pipeline(self, X):
        self.pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    self.build_preprocessor(X)
                ),
                (
                    "model",
                    self.build_estimator()
                ),
            ]
        )

    def train(self, X_train, y_train):
        self.build_pipeline(X_train)
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError(
                "El modelo todavía no ha sido entrenado."
            )

        return self.pipeline.predict(X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        report_text = classification_report(
            y_test,
            predictions,
            zero_division=0
        )

        report_dict = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0
        )

        print(f"\n{'=' * 60}")
        print(f"MODELO: {self.name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(report_text)

        return {
            "accuracy": float(accuracy),
            "classification_report": report_dict,
        }

    def save(self, output_dir):
        if self.pipeline is None:
            raise RuntimeError(
                "No se puede guardar un modelo sin entrenar."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = (
            output_dir /
            f"{self.name}.joblib"
        )

        joblib.dump(
            self.pipeline,
            model_path
        )

        print(
            f"Modelo guardado en: {model_path}"
        )

        return model_path


class LogisticRegressionModel(BasePenguinModel):

    @property
    def name(self):
        return "logistic_regression"

    def build_estimator(self):
        return LogisticRegression(
            max_iter=2000,
            random_state=self.random_state,
        )


class RandomForestModel(BasePenguinModel):

    @property
    def name(self):
        return "random_forest"

    def build_estimator(self):
        return RandomForestClassifier(
            n_estimators=300,
            random_state=self.random_state,
            class_weight="balanced",
        )


MODEL_REGISTRY = {
    "logistic": LogisticRegressionModel,
    "random_forest": RandomForestModel,
}


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    if TARGET not in df.columns:
        raise ValueError(
            f"No se encontró la variable objetivo "
            f"'{TARGET}'. "
            f"Columnas disponibles: "
            f"{df.columns.tolist()}"
        )

    columns_to_drop = [
        col
        for col in DROP_COLUMNS
        if col in df.columns
    ]

    df = df.drop(
        columns=columns_to_drop
    )

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]

    return X, y


def get_models(model_name, random_state):
    if model_name == "all":
        return [
            model_class(
                random_state=random_state
            )
            for model_class
            in MODEL_REGISTRY.values()
        ]

    model_class = MODEL_REGISTRY[
        model_name
    ]

    return [
        model_class(
            random_state=random_state
        )
    ]


def main(
    csv_path,
    output_dir,
    model_name,
    test_size,
    random_state
):
    X, y = load_dataset(csv_path)

    print(f"Dataset: {csv_path}")
    print(f"Registros: {len(X)}")
    print(
        f"Características: "
        f"{X.columns.tolist()}"
    )
    print(
        f"Clases: "
        f"{sorted(y.unique().tolist())}"
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    )

    models = get_models(
        model_name=model_name,
        random_state=random_state
    )

    results = {}

    for model in models:
        model.train(
            X_train,
            y_train
        )

        metrics = model.evaluate(
            X_test,
            y_test
        )

        model_path = model.save(
            output_dir
        )

        metrics["model_path"] = str(
            model_path
        )

        results[
            model.name
        ] = metrics

    if len(results) > 1:
        best_model = max(
            results,
            key=lambda name: (
                results[name]["accuracy"]
            )
        )

        results[
            "best_model"
        ] = best_model

        print(f"\n{'=' * 60}")
        print(
            f"Mejor modelo: "
            f"{best_model}"
        )
        print(
            "Accuracy: "
            f"{results[best_model]['accuracy']:.4f}"
        )

    output_dir = Path(
        output_dir
    )

    metrics_path = (
        output_dir /
        "metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"Métricas guardadas en: "
        f"{metrics_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Entrena modelos para clasificar "
            "especies de pingüinos."
        )
    )

    parser.add_argument(
        "--csv",
        default="penguins.csv",
        help="Ruta al archivo CSV."
    )

    parser.add_argument(
        "--output",
        default="models",
        help=(
            "Carpeta donde se guardarán "
            "los modelos."
        )
    )

    parser.add_argument(
        "--model",
        choices=[
            "logistic",
            "random_forest",
            "all",
        ],
        default="all",
        help=(
            "Modelo que se desea entrenar."
        )
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help=(
            "Proporción usada para prueba."
        )
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria."
    )

    args = parser.parse_args()

    main(
        csv_path=args.csv,
        output_dir=args.output,
        model_name=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
    )