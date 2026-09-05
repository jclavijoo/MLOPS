import shutil
import schedule
import time
from pathlib import Path

MODELS_DIR = Path("./eq_models")
SHARED_DIR = Path("./modelos_globales")

models_copied = []
all_models = []
model_index = 0

def initialize_models():
    """Obtiene lista de todos los modelos al inicio"""
    global all_models
    
    all_models = sorted(list(MODELS_DIR.glob("*.pkl")))
    
    if not all_models:
        print("❌ No hay modelos en ./eq_models/")
        return False
    
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Encontrados {len(all_models)} modelos en eq_models/")
    return True

def add_model_incrementally():
    """Cada 20 segundos agrega un modelo nuevo"""
    global model_index, models_copied, all_models
    
    try:
        if model_index >= len(all_models):
            model_index = 0
            print("Enviando")
        
        current_model = all_models[model_index]
        
        dest = SHARED_DIR / current_model.name
        shutil.copy2(current_model, dest)
        
        if current_model.name not in models_copied:
            models_copied.append(current_model.name)
        
        print(f"Agregado: {current_model.name}")
        print(f"Almacenados ({len(models_copied)}/{len(all_models)}): {models_copied}")
        
        dest_current = SHARED_DIR / "current_model.pkl"
        shutil.copy2(current_model, dest_current)
        
        print(f"Usando ahora: {current_model.name}")
        
        model_index += 1
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Flujo principal"""
    
    if not initialize_models():
        return
    
    schedule.every(60).seconds.do(add_model_incrementally)
    
    print(f"Total de modelos a copiar: {len(all_models)}")
    
    add_model_incrementally()
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()