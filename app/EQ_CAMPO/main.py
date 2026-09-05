import gc
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

app = FastAPI(
    title="EQ_CAMPO",
    description="Servicio de predicciones con modelos rotantes",
    version="1.0.0"
)

current_model = None
current_model_name = "unknown"

# RUTA UNIFICADA CON ENVIAR_MODELO.PY
MODELS_DIR = Path("/home/estudiante/MLOPS/app/modelos_globales")

# ============ MODELOS PYDANTIC ============

class ModelInfo(BaseModel):
    name: str
    size_kb: float
    type: str

class StatusResponse(BaseModel):
    service: str
    model_loaded: bool
    available_models: list
    model_size: int

# ============ FUNCIONES ============

def load_model(model_name: str = "current_model.pkl") -> bool:
    global current_model, current_model_name
    
    try:
        model_path = MODELS_DIR / model_name
        
        if not model_path.exists():
            print(f" {model_path} NO EXISTE")
            current_model = None
            current_model_name = "unknown"
            return False
        
        if current_model is not None:
            del current_model
            current_model = None
            gc.collect()
        
        with open(model_path, "rb") as f:
            current_model = joblib.load(f)
            
        current_model_name = model_path.name
        model_type = type(current_model).__name__
        size = model_path.stat().st_size / 1024
        
        print(f" Cargado: {model_path.name} | Tipo: {model_type} | {size:.2f} KB")
        return True
        
    except Exception as e:
        print(f" Error al cargar el modelo: {e}")
        current_model = None
        return False

def list_all_models() -> list:
    try:
        if not MODELS_DIR.exists():
            return []
        
        models = []
        for model_file in sorted(MODELS_DIR.glob("*.pkl")):
            size_kb = model_file.stat().st_size / 1024
            models.append({
                "name": model_file.name,
                "size_kb": round(size_kb, 2),
                "type": "current" if model_file.name == "current_model.pkl" else "backup"
            })
        
        return models
    except Exception as e:
        print(f" Error al listar modelos: {e}")
        return []

# ============ EVENTOS ============

@app.on_event("startup")
async def startup():
    print(" EQ_CAMPO iniciado")
    print(f" Buscando en: {MODELS_DIR.absolute()}")
    load_model("current_model.pkl")

# ============ ENDPOINTS ============

@app.get("/models", response_model=list)
def get_models():
    return list_all_models()

@app.get("/predict")
def predict():
    success = load_model("current_model.pkl")
    
    if not success or current_model is None:
        return {"error": "Modelo no cargado o no encontrado"}
    
    try:
        model_path = MODELS_DIR / "current_model.pkl"
        
        return {
            "status": "success",
            "model_file": current_model_name,
            "model_type": type(current_model).__name__,
            "model_size_kb": round(model_path.stat().st_size / 1024, 2),
            "prediction": "resultado"
        }
    except Exception as e:
        return {"error": str(e)}