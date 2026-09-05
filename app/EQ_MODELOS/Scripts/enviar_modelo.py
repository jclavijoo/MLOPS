import shutil
import schedule
import time
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("./eq_models")
SHARED_DIR = Path("./modelos_globales")

def send_model():
    """Envía uno de los 3 modelos a la carpeta compartida cada minuto"""
    try:
        models = list(MODELS_DIR.glob("*.pkl"))
        
        if not models:
            logger.error("❌ No hay modelos en ./eq_models/")
            return
        
        selected = random.choice(models)
        SHARED_DIR.mkdir(parents=True, exist_ok=True)
        
        dest = SHARED_DIR / "current_model.pkl"
        shutil.copy2(selected, dest)
        
        logger.info(f"📤 Enviado: {selected.name} → modelos_globales/current_model.pkl")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")

def schedule_sending():
    """Envía modelo cada minuto"""
    schedule.every(1).minute.do(send_model)
    
    logger.info("⏰ EQ_MODELOS - Enviando modelo cada 1 minuto")
    
    # Enviar uno al iniciar
    send_model()
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    schedule_sending()