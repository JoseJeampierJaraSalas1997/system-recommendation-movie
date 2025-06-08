import numpy as np
from src.data_preprocessing import MovieDataPreprocessor
from src.evaluation import ModelEvaluator
from tensorflow.keras.models import load_model

# Rutas de modelos y datos (ajusta según tus rutas)
CNN_MODEL_PATH = 'data/models/cnn_movie_model_best.h5'
MLP_MODEL_PATH = 'data/models/mlp_movie_model_best.h5'

# Cargar modelos
print("📥 Cargando modelos...")
cnn = load_model(CNN_MODEL_PATH)
mlp = load_model(MLP_MODEL_PATH)

# Recrear el preprocesador
DATA_PATH = "data/raw/Multi_Label_dataset"
IMG_SIZE = (224, 224)

print("🔄 Inicializando preprocesador...")
preprocessor = MovieDataPreprocessor(DATA_PATH, IMG_SIZE)

# Cargar datos
print("📂 Cargando datos...")
X, y_labels = preprocessor.load_data()

# Dividir datos
print("✂️  Dividiendo datos...")
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.create_data_splits(
    X, y_labels, test_size=0.2, val_size=0.1
)

# --- 🔁 Crear evaluator con label_encoder válido ---
# Si preprocessor.label_encoder existe (como LabelBinarizer), úsalo:
try:
    label_encoder = preprocessor.label_encoder 
except AttributeError:
    # Si no, usa una lista o pd.Index para crear uno dummy
    from sklearn.preprocessing import LabelBinarizer

    class DummyLabelEncoder:
        def __init__(self, classes):
            self.classes_ = classes

        def inverse_transform(self, y):
            lb = LabelBinarizer()
            lb.fit(self.classes_)
            return lb.inverse_transform(y)

        def transform(self, y):
            lb = LabelBinarizer()
            lb.fit(self.classes_)
            return lb.transform(y)

    

evaluator = ModelEvaluator(label_encoder=label_encoder)

# Evaluar CNN
print("\n📊 EVALUANDO CNN...")
try:
    cnn_metrics, cnn_report, cnn_pred, cnn_proba = evaluator.evaluate_model(cnn, X_test, y_test)
    print("🎯 Métricas CNN:")
    for metric, value in cnn_metrics.items():
        print(f"  {metric.title()}: {value:.4f}")
    evaluator.plot_confusion_matrix(y_test.argmax(axis=1), cnn_pred, 'CNN', 'results/plots')
    evaluator.plot_classification_report(cnn_report, 'CNN', 'results/plots')
except Exception as e:
    print(f"❌ Error evaluando CNN: {str(e)}")

# Evaluar MLP
print("\n📊 EVALUANDO MLP...")
try:
    mlp_metrics, mlp_report, mlp_pred, mlp_proba = evaluator.evaluate_model(mlp, X_test, y_test)
    print("🎯 Métricas MLP:")
    for metric, value in mlp_metrics.items():
        print(f"  {metric.title()}: {value:.4f}")
    evaluator.plot_confusion_matrix(y_test.argmax(axis=1), mlp_pred, 'MLP', 'results/plots')
    evaluator.plot_classification_report(mlp_report, 'MLP', 'results/plots')
except Exception as e:
    print(f"❌ Error evaluando MLP: {str(e)}")

print("\n✅ Evaluación completada.")