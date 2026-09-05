from fastapi import FastAPI
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
current_model = None

def load_model():
    """Carga el modelo actual de la carpeta compartida"""
    global current_model
    
    try:
        model_path = Path("./modelos_globales/current_model.pkl")
        
        if model_path.exists():
            current_model = joblib.load(model_path)
            logger.info(f"Modelo cargado: {model_path}")
            return True
        else:
            logger.warning("current_model.pkl no encontrado")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")
        return False

@app.on_event("startup")
async def startup():
    logger.info("EQ_CAMPO iniciado")
    load_model()

@app.get("/")
def root():
    load_model()
    return {
        "service": "EQ_CAMPO",
        "model_loaded": current_model is not None
    }

@app.get("/predict")
def predict():
    """Predicción con el modelo actual"""
    load_model()
    
    if current_model is None:
        return {"error": "Model not loaded", "status": "error"}
    
    try:
        return {
            "status": "success",
            "model_type": type(current_model).__name__
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

@app.get("/status")
def status():
    """Ver estado del modelo actual"""
    model_path = Path("./modelos_globales/current_model.pkl")
    
    return {
        "service": "EQ_CAMPO",
        "model_loaded": current_model is not None,
        "model_exists": model_path.exists(),
        "model_size": model_path.stat().st_size if model_path.exists() else 0
    }