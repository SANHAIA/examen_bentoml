import bentoml
from bentoml.io import JSON
import numpy as np
import joblib

model_runner = bentoml.sklearn.get("admissions_model:latest").to_runner()

scaler = joblib.load("models/scaler.joblib")

svc = bentoml.Service("admissions_service", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def predict(input_data):
    # Créer un tableau avec les 7 colonnes fournies
    input_array = np.array([[
        input_data["GRE Score"],
        input_data["TOEFL Score"],
        input_data["University Rating"],
        input_data["SOP"],
        input_data["LOR"],
        input_data["CGPA"],
        input_data["Research"]
    ]])

    print("Forme des données envoyées au modèle :", input_array.shape)
    input_array_scaled = scaler.transform(input_array)
    # Effectuer la prédiction
    prediction = model_runner.run(input_array)
    
    prediction_clipped = np.clip(prediction, 0, 1)
    
    return {"prediction": prediction.tolist()}


