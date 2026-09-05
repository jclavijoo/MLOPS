from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import importlib
import sys
from pathlib import Path

app = FastAPI(
    title="EQ_CAMPO",
    description="Servicio de predicciones con modelos rotantes",
    version="1.0.0"
)

current_model = None
MODELS_DIR = Path("./modelos_globales")

# ============ MODELOS PYDANTIC ============

class ModelInfo(BaseModel):
    """Información de un modelo"""
    name: str
    size_kb: float
    type: str

class PredictionResponse(BaseModel):
    """Respuesta de predicción"""
    status: str
    model_name: str
    model_type: str
    prediction: str = "resultado"

class StatusResponse(BaseModel):
    """Estado del servicio"""
    service: str
    model_loaded: bool
    current_model: str
    available_models: list
    model_size: int

# ============ FUNCIONES ============

def load_model(model_name: str = "current_model.pkl"):
    """Carga un modelo específico - FUERZA RECARGA DEL DISCO"""
    global current_model
    
    try:
        model_path = MODELS_DIR / model_name
        
        if model_path.exists():
            # Obtener timestamp ANTES
            timestamp_before = model_path.stat().st_mtime
            
            # Eliminar objeto anterior
            current_model = None
            
            # Cargar NUEVO del disco
            current_model = joblib.load(model_path)
            
            # Obtener timestamp DESPUÉS
            timestamp_after = model_path.stat().st_mtime
            
            # Información del modelo
            model_type = type(current_model).__name__
            model_size = model_path.stat().st_size / 1024
            
            print(f" Modelo cargado: {model_name}")
            print(f" Tipo: {model_type}")
            print(f" Tamaño: {model_size:.2f} KB")
            print(f" Timestamp: {timestamp_before}")
            
            return True
        else:
            print(f"{model_name} no encontrado")
            return False
            
    except Exception as e:
        print(f"❌ Error cargando {model_name}: {e}")
        return False

def list_all_models() -> list:
    """Lista todos los modelos disponibles"""
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
        print(f"Error listando modelos: {e}")
        return []

# ============ EVENTOS ============

@app.on_event("startup")
async def startup():
    print("EQ_CAMPO iniciado")
    print(f"Carpeta de modelos: {MODELS_DIR.absolute()}")
    
    # Cargar modelo actual
    load_model("current_model.pkl")
    
    # Listar modelos disponibles
    models = list_all_models()
    print(f"Modelos disponibles: {len(models)}")
    for m in models:
        print(f"   • {m['name']} ({m['size_kb']} KB) - [{m['type']}]")

# ============ ENDPOINTS ============

@app.get("/models", tags=["Modelos"], response_model=list, summary="Listar todos los modelos disponibles")
def get_models():
    models = list_all_models()
    return models

@app.get("/current-model", tags=["Modelos"], summary="Ver modelo actual")
def get_current_model():
    model_path = MODELS_DIR / "current_model.pkl"
    
    if model_path.exists():
        return {
            "model_name": "current_model.pkl",
            "type": type(current_model).__name__ if current_model else "Unknown",
            "size_kb": round(model_path.stat().st_size / 1024, 2),
            "loaded": current_model is not None
        }
    else:
        return {
            "error": "current_model.pkl no encontrado",
            "status": "error"
        }

@app.get("/predict", tags=["Predicciones"], summary="Hacer predicción con modelo actual")
def predict():
    # Recargar por si cambió
    load_model("current_model.pkl")
    
    if current_model is None:
        return {
            "status": "error",
            "error": "Modelo no cargado",
            "model_loaded": False
        }
    
    try:
        model_path = MODELS_DIR / "current_model.pkl"
        model_type = type(current_model).__name__
        model_module = current_model.__class__.__module__
        
        print(f"DEBUG - Modelo actual: {model_type}")
        print(f"DEBUG - Módulo: {model_module}")
        print(f"DEBUG - Archivo: {model_path.name}")
        
        return {
            "status": "success",
            "model_name": model_path.name,
            "model_type": model_type,
            "model_module": model_module,
            "prediction": "resultado ejemplo",
            "model_size_kb": round(model_path.stat().st_size / 1024, 2) if model_path.exists() else 0
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/status", tags=["Info"], summary="Estado general del servicio", response_model=StatusResponse)
def status():
    model_path = MODELS_DIR / "current_model.pkl"
    available_models = [m["name"] for m in list_all_models()]
    
    return StatusResponse(
        service="EQ_CAMPO",
        model_loaded=current_model is not None,
        current_model="current_model.pkl",
        available_models=available_models,
        model_size=model_path.stat().st_size if model_path.exists() else 0
    )