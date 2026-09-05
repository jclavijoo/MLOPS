import shutil
import schedule
import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import joblib

MODELS_DIR = Path("./eq_models")
SHARED_DIR = Path("./modelos_globales")

all_models = []
model_index = 0

def create_models():
    """Crea 3 modelos DIFERENTES"""
    print(" Creando 3 modelos DIFERENTES...\n")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Limpiar anteriores
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for f in MODELS_DIR.glob("*.pkl"):
        f.unlink()
    
    # MODELO 1: RandomForest
    model1 = RandomForestClassifier(n_estimators=100, random_state=42)
    model1.fit(X, y)
    joblib.dump(model1, MODELS_DIR / "model_1.pkl")
    size1 = (MODELS_DIR / "model_1.pkl").stat().st_size / 1024
    print(f" model_1.pkl: {type(model1).__name__} ({size1:.2f} KB)")
    
    # MODELO 2: SVC
    model2 = SVC(kernel='rbf', random_state=42)
    model2.fit(X, y)
    joblib.dump(model2, MODELS_DIR / "model_2.pkl")
    size2 = (MODELS_DIR / "model_2.pkl").stat().st_size / 1024
    print(f" model_2.pkl: {type(model2).__name__} ({size2:.2f} KB)")
    
    # MODELO 3: KNeighbors
    model3 = KNeighborsClassifier(n_neighbors=5)
    model3.fit(X, y)
    joblib.dump(model3, MODELS_DIR / "model_3.pkl")
    size3 = (MODELS_DIR / "model_3.pkl").stat().st_size / 1024
    print(f" model_3.pkl: {type(model3).__name__} ({size3:.2f} KB)")
    
    print("\n Verificación final:")
    for name in ["model_1.pkl", "model_2.pkl", "model_3.pkl"]:
        m = joblib.load(MODELS_DIR / name)
        print(f"   {name}: {type(m).__name__}")

def initialize_models():
    """Obtiene lista de todos los modelos"""
    global all_models
    
    all_models = sorted(list(MODELS_DIR.glob("*.pkl")))
    
    if not all_models:
        print(" No hay modelos en ./eq_models/")
        return False
    
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n Encontrados {len(all_models)} modelos")
    print(f" Rotando cada 10 segundos")
    print(f"  Borrando cada 8 segundos\n")
    return True

def add_model_incrementally():
    """Cada 10 segundos agrega un modelo nuevo"""
    global model_index, all_models
    
    try:
        # PRIMERO: Borrar el anterior
        dest_current = SHARED_DIR / "current_model.pkl"
        if dest_current.exists():
            dest_current.unlink()
            print(f" Eliminado: current_model.pkl")
        
        # LUEGO: Crear el nuevo
        if model_index >= len(all_models):
            model_index = 0
            print("Ciclo completado\n")
        
        current_model = all_models[model_index]
        
        dest = SHARED_DIR / current_model.name
        shutil.copy2(current_model, dest)
        
        dest_current = SHARED_DIR / "current_model.pkl"
        shutil.copy2(current_model, dest_current)
        
        model_type = joblib.load(current_model).__class__.__name__
        size = current_model.stat().st_size / 1024
        
        print(f"[{model_index + 1}/{len(all_models)}] {current_model.name}: {model_type} ({size:.2f} KB)")
        
        model_index += 1
        
    except Exception as e:
        print(f" Error: {e}")

def main():
    create_models()
    if not initialize_models():
        return
    schedule.every(10).seconds.do(add_model_incrementally)
    
    add_model_incrementally()
    
    while True:
        schedule.run_pending()
        time.sleep(1)
        
    except Exception as e:
        print(f"Error: {e}")