import shutil
import schedule
import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import joblib

# Carpeta local para los modelos base
MODELS_DIR = Path("/home/estudiante/MLOPS/app/eq_models")
# Carpeta compartida donde se publica el modelo activo
SHARED_DIR = Path("/home/estudiante/MLOPS/app/modelos_globales")

all_models = []
model_index = 0

def create_models():
    """Crea 3 modelos DIFERENTES"""
    print(" Creando 3 modelos DIFERENTES...\n")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for f in MODELS_DIR.glob("*.pkl"):
        f.unlink()
    
    # MODELO 1: RandomForest
    model1 = RandomForestClassifier(n_estimators=100, random_state=42)
    model1.fit(X, y)
    joblib.dump(model1, MODELS_DIR / "model_1.pkl")
    
    # MODELO 2: SVC
    model2 = SVC(kernel='rbf', random_state=42)
    model2.fit(X, y)
    joblib.dump(model2, MODELS_DIR / "model_2.pkl")
    
    # MODELO 3: KNeighbors
    model3 = KNeighborsClassifier(n_neighbors=5)
    model3.fit(X, y)
    joblib.dump(model3, MODELS_DIR / "model_3.pkl")

def initialize_models():
    global all_models
    all_models = sorted(list(MODELS_DIR.glob("*.pkl")))
    
    if not all_models:
        print(" No hay modelos en eq_models")
        return False
    
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n Encontrados {len(all_models)} modelos")
    return True

def add_model_incrementally():
    global model_index, all_models
    
    try:
        if model_index >= len(all_models):
            model_index = 0
            print("Ciclo completado\n")
        
        current_model = all_models[model_index]
        dest_current = SHARED_DIR / "current_model.pkl"
        
        # Copia directa atómica
        shutil.copy2(current_model, dest_current)
        
        model_type = joblib.load(current_model).__class__.__name__
        size = current_model.stat().st_size / 1024
        
        print(f"[{model_index + 1}/{len(all_models)}] Copiado {current_model.name} -> current_model.pkl | Tipo: {model_type} ({size:.2f} KB)")
        
        model_index += 1
        
    except Exception as e:
        print(f" Error al rotar modelo: {e}")

def main():
    try:
        create_models()
        if not initialize_models():
            return

        schedule.every(5).seconds.do(add_model_incrementally)
        add_model_incrementally()

        while True:
            schedule.run_pending()
            time.sleep(1)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()