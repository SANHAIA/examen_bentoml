import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def prepare_data(input_path, output_dir):
   
    # Charger les données
    data = pd.read_csv(input_path)
    
    # Supprimer les lignes avec des valeurs manquantes
    data = data.dropna()

    if "Serial No." in data.columns:
        data = data.drop(columns=["Serial No."])  # Suppression d'une colonne non pertinente

    # Identifier la cible et les features
    X = data.drop(columns=["Chance of Admit "])
    y = data["Chance of Admit "]

    # Standardiser les variables explicatives (si elles sont quantitatives)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Diviser les données en jeux d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Sauvegarder les fichiers dans le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X_train).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print("Données préparées et sauvegardées dans :", output_dir)

if __name__ == "__main__":
    prepare_data("../data/raw/admission.csv", "../data/processed")
