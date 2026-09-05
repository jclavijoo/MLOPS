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
    """Carga un modelo - FUERZA RECARGA"""
    global current_model, current_model_name
    
    try:
        model_path = MODELS_DIR / model_name
        
        if not model_path.exists():
            print(f" {model_path} NO EXISTE")
            print(f"   Buscando en: {MODELS_DIR.absolute()}")
            print(f"   Contenido: {list(MODELS_DIR.glob('*'))}")
            return False
        
        # Limpiar
        current_model = None
        
        # Cargar
        current_model = joblib.load(model_path)
        current_model_name = model_path.name
        
        model_type = type(current_model).__name__
        size = model_path.stat().st_size / 1024
        
        print(f" Cargado: {model_path.name} | {model_type} | {size:.2f}KB")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
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

# ============ ENDPOINTS ============

@app.get("/models", response_model=list)
def get_models():
    return list_all_models()

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

@app.on_event("startup")
async def startup():
    print("EQ_CAMPO iniciado")
    print(f" Buscando en: {MODELS_DIR.absolute()}")
    print(f"  Existe carpeta: {MODELS_DIR.exists()}")
    
    # Listar archivos
    if MODELS_DIR.exists():
        files = list(MODELS_DIR.glob("*.pkl"))
        print(f"   Archivos: {[f.name for f in files]}")
    
    load_model("current_model.pkl")