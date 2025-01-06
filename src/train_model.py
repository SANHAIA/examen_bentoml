from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import bentoml
from sklearn.preprocessing import StandardScaler
import joblib

def train_and_evaluate_model(X_train_path, y_train_path, X_test_path, y_test_path, model_store_path):
    """
    Entraîne un modèle de régression linéaire, évalue ses performances, et le sauvegarde.
    """
    # Charger les données
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()  # Convertir en tableau 1D
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    joblib.dump(scaler, "models/scaler.joblib")
    
    # Initialiser et entraîner le modèle
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    model.fit(X_train, y_train)

    # Évaluation sur les données de test
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Affichage des métriques d'évaluation
    print("Forme de X_train :", X_train.shape)
    print(f"Performance du modèle :")
    print(f"  - Mean Squared Error (MSE): {mse:.4f}")
    print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  - Coefficient of Determination (R2 Score): {r2:.4f}")

    # Sauvegarde du modèle dans BentoML
    bentoml.sklearn.save_model(model_store_path, model)
    print(f"Modèle sauvegardé dans le Model Store de BentoML sous le nom : {model_store_path}")

if __name__ == "__main__":
    train_and_evaluate_model(
        X_train_path="data/processed/X_train.csv",
        y_train_path="data/processed/y_train.csv",
        X_test_path="data/processed/X_test.csv",
        y_test_path="data/processed/y_test.csv",
        model_store_path="admissions_model"
    )
