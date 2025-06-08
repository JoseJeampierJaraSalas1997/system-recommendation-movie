import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import plot_model
import json

def create_directories():
    """Crear estructura de directorios del proyecto"""
    directories = [
        'data/raw', 'data/processed', 'data/models',
        'results/plots', 'results/metrics', 'results/report',
        'notebooks', 'src'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Estructura de directorios creada exitosamente")

def save_training_history(history, model_name, save_path):
    """Guardar historial de entrenamiento"""
    history_dict = {
        'loss': history.history['loss'],
        'accuracy': history.history['accuracy'],
        'val_loss': history.history['val_loss'],
        'val_accuracy': history.history['val_accuracy']
    }
    
    with open(f"{save_path}/{model_name}_history.json", 'w') as f:
        json.dump(history_dict, f, indent=4)

def plot_training_history(history, model_name, save_path):
    """Crear gráficas del historial de entrenamiento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfica de pérdida
    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica de precisión
    ax2.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/{model_name}_training_history.png", dpi=300, bbox_inches='tight')
    plt.show()

def setup_gpu():
    """Configurar GPU si está disponible"""
    import tensorflow as tf
    
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU configurada correctamente")
        except RuntimeError as e:
            print(f"Error configurando GPU: {e}")
    else:
        print("Usando CPU para entrenamiento")