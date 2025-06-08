#!/usr/bin/env python3
"""
Sistema de Recomendación de Películas basado en Imágenes
Proyecto Final - Sistemas de Recomendación
Optimizado para Clasificación Multi-Etiqueta
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar módulos locales
sys.path.append('src')
from utils import create_directories, setup_gpu, plot_training_history, save_training_history
from data_preprocessing import MovieDataPreprocessor
from cnn_model import CNNModel
from mlp_model import MLPModel
from evaluation import ModelEvaluator

def print_data_verification(X, y_labels, preprocessor):
    """Función para verificar la estructura de los datos"""
    print(f"\n🔍 VERIFICACIÓN DE DATOS:")
    print(f"  X shape: {X.shape}")
    print(f"  y_labels shape: {y_labels.shape}")
    print(f"  y_labels dtype: {y_labels.dtype}")
    print(f"  Número de clases: {len(preprocessor.label_columns)}")
    print(f"  Clases: {preprocessor.label_columns}")
    print(f"  Ejemplo de etiquetas (primera muestra): {y_labels[0]}")
    print(f"  Suma de etiquetas por muestra (primeras 5): {y_labels.sum(axis=1)[:5]}")
    print(f"  Rango de valores en etiquetas: [{y_labels.min()}, {y_labels.max()}]")

def verify_generators(train_gen, val_gen):
    """Función para verificar los generadores de datos"""
    print(f"\n🔍 VERIFICACIÓN DE GENERADORES:")
    
    # Verificar generador de entrenamiento
    for batch_x, batch_y in train_gen:
        print(f"  Train Batch X shape: {batch_x.shape}")
        print(f"  Train Batch Y shape: {batch_y.shape}")
        print(f"  Train Batch Y dtype: {batch_y.dtype}")
        print(f"  Train Batch Y ejemplo: {batch_y[0]}")
        print(f"  Train Batch Y suma (primeras 3): {batch_y.sum(axis=1)[:3]}")
        break
    
    # Verificar generador de validación
    for batch_x, batch_y in val_gen:
        print(f"  Val Batch X shape: {batch_x.shape}")
        print(f"  Val Batch Y shape: {batch_y.shape}")
        print(f"  Val Batch Y dtype: {batch_y.dtype}")
        break
def print_model_info(model, model_name):
    """
    Versión simplificada que solo usa métodos básicos de Keras
    """
    print(f"\n🏗️  INFORMACIÓN DEL MODELO {model_name}:")
    
    try:
        total_params = model.count_params()
        print(f"  Total de parámetros: {total_params:,}")
        print(f"  Número de capas: {len(model.layers)}")
        
        # Mostrar resumen básico
        print("\n📋 RESUMEN BÁSICO:")
        model.summary()
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  No se pudo obtener información del modelo")

def main():
    print("="*70)
    print("SISTEMA DE RECOMENDACIÓN DE PELÍCULAS")
    print("Basado en Análisis de Imágenes con Deep Learning")
    print("CLASIFICACIÓN MULTI-ETIQUETA DE GÉNEROS")
    print("="*70)
    
    # Configuración inicial
    print("\n⚙️  CONFIGURACIÓN INICIAL")
    print("-" * 30)
    create_directories()
    setup_gpu()
    
    # Parámetros del proyecto
    DATA_PATH = "data/raw/Multi_Label_dataset"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS_CNN = 100
    EPOCHS_MLP = 150
    
    print(f"📁 Ruta de datos: {DATA_PATH}")
    print(f"🖼️  Tamaño de imagen: {IMG_SIZE}")
    print(f"📦 Tamaño de batch: {BATCH_SIZE}")
    print(f"🔄 Épocas CNN: {EPOCHS_CNN}")
    print(f"🔄 Épocas MLP: {EPOCHS_MLP}")
    
    print("\n" + "="*70)
    print("1. CARGA Y PREPROCESAMIENTO DE DATOS")
    print("="*70)
    
    # Inicializar preprocessor
    preprocessor = MovieDataPreprocessor(DATA_PATH, IMG_SIZE)
    
    # Cargar datos
    print("📥 Cargando imágenes desde:", DATA_PATH)
    X, y_labels = preprocessor.load_data()
    print(f"✅ Imágenes cargadas: {len(X)}")
    
    # Verificar estructura de datos
    print_data_verification(X, y_labels, preprocessor)
    
    # Info dataset
    dataset_info = preprocessor.get_dataset_info(y_labels)
    print(f"\n📊 INFORMACIÓN DEL DATASET:")
    print(f"  Total de muestras: {dataset_info['total_samples']}")
    print(f"  Número de clases: {dataset_info['num_classes']}")
    print(f"  Clases por muestra (promedio): {y_labels.sum(axis=1).mean():.2f}")
    
    print(f"\n📈 Distribución de clases:")
    for clase, count in dataset_info['class_distribution'].items():
        percentage = (count / dataset_info['total_samples']) * 100
        print(f"  {clase}: {count} muestras ({percentage:.1f}%)")
    
    # Dividir datos
    print(f"\n🔄 Dividiendo datos...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.create_data_splits(
        X, y_labels, test_size=0.2, val_size=0.1
    )
    
    print(f"📊 División de datos:")
    print(f"  🎯 Entrenamiento: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  🔍 Validación: {len(X_val)} muestras ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  📋 Prueba: {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")
    
    # Verificar formas después de la división
    print(f"\n🔍 Verificación post-división:")
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Visualizar muestras
    print(f"\n📸 Generando visualizaciones...")
    preprocessor.visualize_samples(X_train, y_train)
    
    # Crear generadores
    print(f"\n⚙️  Creando generadores de datos...")
    train_gen, val_gen = preprocessor.create_data_generators(
        X_train, y_train, X_val, y_val, BATCH_SIZE
    )
    
    # Verificar generadores
    verify_generators(train_gen, val_gen)
    
    print("\n" + "="*70)
    print("2. CONSTRUCCIÓN Y ENTRENAMIENTO DE MODELOS")
    print("="*70)
    
    results = {}
    num_classes = len(preprocessor.label_columns)
    
    # === MODELO CNN ===
    print("\n🔥 CONSTRUYENDO MODELO CNN...")
    print("-" * 40)
    
    cnn_model = CNNModel(input_shape=(*IMG_SIZE, 3), num_classes=num_classes)
    cnn = cnn_model.build_model(architecture='custom')  # Usar arquitectura personalizada
    
    # Mostrar información del modelo
    print_model_info(cnn, 'CNN')
    
    # Mostrar resumen del modelo
    print(f"\n📋 Resumen del modelo CNN:")
    cnn.summary()
    
    # Entrenar CNN
    print(f"\n🚀 ENTRENANDO MODELO CNN...")
    print("-" * 40)
    
    cnn_callbacks = cnn_model.get_callbacks('cnn_movie_model')
    
    try:
        cnn_history = cnn_model.train(
            train_gen, val_gen, 
            epochs=EPOCHS_CNN, 
            callbacks=cnn_callbacks
        )
        
        # Guardar historial
        save_training_history(cnn_history, 'CNN', 'results/metrics')
        plot_training_history(cnn_history, 'CNN', 'results/plots')
        
        print("✅ Entrenamiento CNN completado exitosamente")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento CNN: {str(e)}")
        print("🔍 Verifica la configuración del modelo y los datos")
        return
    
    # === MODELO MLP ===
    print(f"\n🧠 CONSTRUYENDO MODELO MLP...")
    print("-" * 40)
    
    mlp_model = MLPModel(input_shape=(*IMG_SIZE, 3), num_classes=num_classes)
    mlp = mlp_model.build_model(
        hidden_layers=[1024, 512, 256], 
        dropout_rates=[0.5, 0.4, 0.3]
    )
    
    # Mostrar información del modelo
    print_model_info(mlp, 'MLP')
    
    # Mostrar resumen del modelo
    print(f"\n📋 Resumen del modelo MLP:")
    mlp.summary()
    
    # Entrenar MLP
    print(f"\n🚀 ENTRENANDO MODELO MLP...")
    print("-" * 40)
    
    try:
        mlp_history = mlp_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=EPOCHS_MLP,
            batch_size=BATCH_SIZE,
            model_name='mlp_movie_model'
        )
        
        # Guardar historial
        save_training_history(mlp_history, 'MLP', 'results/metrics')
        plot_training_history(mlp_history, 'MLP', 'results/plots')
        
        print("✅ Entrenamiento MLP completado exitosamente")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento MLP: {str(e)}")
        print("🔍 Verifica la configuración del modelo y los datos")
        return
    
    print("\n" + "="*70)
    print("3. EVALUACIÓN DE MODELOS")
    print("="*70)
    
    evaluator = ModelEvaluator(classes=preprocessor.label_columns)
    
    # Evaluar CNN
    print("\n📊 EVALUANDO CNN...")
    print("-" * 30)
    
    try:
        cnn_metrics, cnn_report, cnn_pred, cnn_proba = evaluator.evaluate_model(
            cnn, X_test, y_test
        )
        
        print("🎯 Métricas CNN:")
        for metric, value in cnn_metrics.items():
            print(f"  {metric.title()}: {value:.4f}")
        
        evaluator.plot_confusion_matrix(y_test, cnn_pred, 'CNN', 'results/plots')
        evaluator.plot_classification_report(cnn_report, 'CNN', 'results/plots')
        
        results['CNN'] = {
            'metrics': cnn_metrics,
            'report': cnn_report,
            'predictions': cnn_pred,
            'probabilities': cnn_proba
        }
        
        print("✅ Evaluación CNN completada")
        
    except Exception as e:
        print(f"❌ Error durante la evaluación CNN: {str(e)}")
    
    # Evaluar MLP
    print("\n📊 EVALUANDO MLP...")
    print("-" * 30)
    
    try:
        mlp_metrics, mlp_report, mlp_pred, mlp_proba = evaluator.evaluate_model(
            mlp, X_test, y_test
        )
        
        print("🎯 Métricas MLP:")
        for metric, value in mlp_metrics.items():
            print(f"  {metric.title()}: {value:.4f}")
        
        evaluator.plot_confusion_matrix(y_test, mlp_pred, 'MLP', 'results/plots')
        evaluator.plot_classification_report(mlp_report, 'MLP', 'results/plots')
        
        results['MLP'] = {
            'metrics': mlp_metrics,
            'report': mlp_report,
            'predictions': mlp_pred,
            'probabilities': mlp_proba
        }
        
        print("✅ Evaluación MLP completada")
        
    except Exception as e:
        print(f"❌ Error durante la evaluación MLP: {str(e)}")
    
    print("\n" + "="*70)
    print("4. COMPARACIÓN Y ANÁLISIS DE RESULTADOS")
    print("="*70)
    
    if len(results) >= 2:
        comparison_df = evaluator.compare_models(results, 'results/plots')
        print("\n📊 Comparación de modelos:")
        print(comparison_df.to_string(index=False))
        
        evaluator.save_metrics_to_csv(results, 'results/metrics')
        
        # Determinar mejor modelo
        cnn_acc = results['CNN']['metrics']['accuracy'] if 'CNN' in results else 0
        mlp_acc = results['MLP']['metrics']['accuracy'] if 'MLP' in results else 0
        mejor_modelo = 'CNN' if cnn_acc > mlp_acc else 'MLP'
        
        print(f"\n🏆 Mejor modelo: {mejor_modelo}")
        print(f"  CNN Accuracy: {cnn_acc:.4f}")
        print(f"  MLP Accuracy: {mlp_acc:.4f}")
        print(f"  Diferencia: {abs(cnn_acc - mlp_acc):.4f}")
    
    print("\n" + "="*70)
    print("5. GUARDADO DE MODELOS")
    print("="*70)
    
    try:
        # Crear directorio si no existe
        os.makedirs('data/models', exist_ok=True)
        
        # Guardar modelos
        cnn_model.save_model('data/models/cnn_final_model.h5')
        mlp.save('data/models/mlp_final_model.h5')
        
        print("✅ Modelos guardados exitosamente")
        print("  📁 CNN: data/models/cnn_final_model.h5")
        print("  📁 MLP: data/models/mlp_final_model.h5")
        
    except Exception as e:
        print(f"❌ Error al guardar modelos: {str(e)}")
    
    print("\n" + "="*70)
    print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*70)
    
    # Resumen final
    print(f"\n📈 RESUMEN FINAL DEL PROYECTO:")
    print(f"  📊 Dataset: {dataset_info['total_samples']} imágenes")
    print(f"  🏷️  Clases: {dataset_info['num_classes']} géneros")
    print(f"  🎭 Tipo: Clasificación Multi-Etiqueta")
    print(f"  🖼️  Resolución: {IMG_SIZE[0]}x{IMG_SIZE[1]} píxeles")
    print(f"  📦 Batch size: {BATCH_SIZE}")
    
    if 'CNN' in results and 'MLP' in results:
        print(f"\n🎯 RENDIMIENTO DE MODELOS:")
        print(f"  CNN Accuracy: {results['CNN']['metrics']['accuracy']:.4f}")
        print(f"  MLP Accuracy: {results['MLP']['metrics']['accuracy']:.4f}")
        print(f"  🏆 Mejor modelo: {mejor_modelo}")
    
    print(f"\n📁 ARCHIVOS GENERADOS:")
    print(f"  📊 Gráficas: results/plots/")
    print(f"  📈 Métricas: results/metrics/")
    print(f"  🤖 Modelos: data/models/")
    
    print(f"\n💡 PRÓXIMOS PASOS:")
    print(f"  1. Analizar las métricas de evaluación")
    print(f"  2. Revisar las matrices de confusión")
    print(f"  3. Implementar sistema de recomendación")
    print(f"  4. Optimizar hiperparámetros si es necesario")
    
    print(f"\n🎊 ¡PROYECTO COMPLETADO CON ÉXITO!")

if __name__ == "__main__":
    main()