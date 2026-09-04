# Configuración de entorno FastAPI para la API de clasificación de pingüinos

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import joblib
import pandas as pd

# Variable global para mantener el modelo actual
current_model = None


# Carga de modelo por defecto al iniciar la API
@asynccontextmanager
async def lifespan(app: FastAPI):
    global current_model
    try:
        current_model = joblib.load("models/logistic_regression.joblib")
        print("Modelo por defecto (Logistic Regression) cargado correctamente.")
    except Exception as e:
        print(f"Error al cargar el modelo inicial: {e}")
    yield

# Creación e inicialización de la instancia de FastAPI
app = FastAPI(
    title="Penguins API",
    description="API para clasificación de especies de pingüinos",
    version="1.0.0",
    lifespan=lifespan,
)


# =========================
# ESQUEMAS DE ENTRADA
# =========================

# Elección de modelo: 1 para Regresión Logística, 2 para Random Forest
class ModelChoice(BaseModel):
    model_name: int  # 1: Logistic Regression, 2: Random Forest

# Esquema de entrada para la predicción de especies de pingüinos
class PenguinInput(BaseModel):
    island: str
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str
    year: int

    # Normalizar en minúsculas para coincidir con el dataset de entrenamiento
    @field_validator("sex", "island")
    @classmethod
    def normalize_strings(cls, v: str) -> str:
        return v.strip().lower() if v else v


# =========================
# ENDPOINTS
# =========================

# Endpoint raíz para verificar el estado de la API
@app.get("/")
async def root():
    return {"status": "API funcionando"}

# Endpoint para realizar predicciones de especies de pingüinos
@app.post("/Predicción")
async def predict(data: PenguinInput):
    global current_model

    if current_model is None:
        raise HTTPException(
            status_code=500,
            detail="El modelo no ha sido inicializado correctamente.",
        )

    # Convertir Pydantic model a DataFrame
    input_dict = data.model_dump()

    # Formatear 'island' a formato Title Case ('Biscoe', 'Dream', 'Torgersen') como en el CSV original
    input_dict["island"] = input_dict["island"].title()

    input_data = pd.DataFrame([input_dict])

    prediction = current_model.predict(input_data)

    return {"prediction": str(prediction[0])}

# Endpoint para cambiar el modelo actual entre Regresión Logística y Random Forest
@app.post("/Modelos")
async def change_model(data: ModelChoice):
    global current_model

    if data.model_name == 1:
        current_model = joblib.load("models/logistic_regression.joblib")
        nombre = "Logistic Regression"
    elif data.model_name == 2:
        current_model = joblib.load("models/random_forest.joblib")
        nombre = "Random Forest"
    else:
        raise HTTPException(
            status_code=400,
            detail="Opción inválida. Usa 1 para Logistic Regression o 2 para Random Forest.",
        )

    return {"mensaje": f"Modelo cambiado con éxito a {nombre}"}