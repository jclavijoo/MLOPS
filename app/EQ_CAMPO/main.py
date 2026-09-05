from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import sys
from pathlib import Path

app = FastAPI(
    title="EQ_CAMPO",
    description="Servicio de predicciones con modelos rotantes",
    version="1.0.0"
)

current_model = None
current_model_name = "unknown"
MODELS_DIR = Path("./modelos_globales")

# ============ MODELOS PYDANTIC ============

class ModelInfo(BaseModel):
    name: str
    size_kb: float
    type: str

class StatusResponse(BaseModel):
    service: str
    model_loaded: bool
    current_model: str
    available_models: list
    model_size: int

# ============ FUNCIONES ============

def load_model(model_name: str = "current_model.pkl"):
    global current_model, current_model_name
    
    try:
        model_path = MODELS_DIR / model_name
        
        if model_path.exists():
            current_model = None
            current_model = joblib.load(model_path)
            current_model_name = model_path.name
            
            model_type = type(current_model).__name__
            model_size = model_path.stat().st_size / 1024
            
            print(f" Modelo cargado: {model_path.name}")
            print(f" Tipo: {model_type}")
            print(f" Tamaño: {model_size:.2f} KB")
            
            return True
        else:
            print(f"{model_name} no encontrado")
            return False
            
    except Exception as e:
        print(f" Error: {e}")
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
        print(f"❌ Error: {e}")
        return []

# ============ EVENTOS ============

@app.on_event("startup")
async def startup():
    print(" EQ_CAMPO iniciado")
    print(f" Carpeta: {MODELS_DIR.absolute()}")
    load_model("current_model.pkl")
    
    models = list_all_models()
    print(f"Modelos: {len(models)}")
    for m in models:
        print(f"   • {m['name']} ({m['size_kb']} KB)")

# ============ ENDPOINTS ============

@app.get("/models", response_model=list)
def get_models():
    return list_all_models()

@app.get("/current-model")
def get_current_model():
    model_path = MODELS_DIR / "current_model.pkl"
    
    if model_path.exists():
        return {
            "model_file": current_model_name,
            "model_type": type(current_model).__name__ if current_model else "Unknown",
            "size_kb": round(model_path.stat().st_size / 1024, 2),
            "loaded": current_model is not None
        }
    else:
        return {"error": "No encontrado"}

@app.get("/predict")
def predict():
    load_model("current_model.pkl")
    
    if current_model is None:
        return {"error": "No cargado"}
    
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

@app.get("/status", response_model=StatusResponse)
def status():
    model_path = MODELS_DIR / "current_model.pkl"
    available_models = [m["name"] for m in list_all_models()]
    
    return StatusResponse(
        service="EQ_CAMPO",
        model_loaded=current_model is not None,
        current_model=current_model_name,
        available_models=available_models,
        model_size=model_path.stat().st_size if model_path.exists() else 0
    )