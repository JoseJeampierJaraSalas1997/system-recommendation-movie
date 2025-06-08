import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense,
    Dropout, BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import os

def print_model_info(model, model_name):
    """
    Versión simplificada que solo usa métodos básicos de Keras
    """
    print(f"\n🏗️  INFORMACIÓN DEL MODELO {model_name}:")
    
class CNNModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=25):
        """
        CNN Model para clasificación multi-etiqueta de géneros de películas
        
        Args:
            input_shape: Forma de las imágenes de entrada (height, width, channels)
            num_classes: Número de géneros (25 en tu dataset)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self, architecture='custom'):
        """
        Construye el modelo CNN para clasificación multi-etiqueta
        
        Args:
            architecture: Tipo de arquitectura ('custom', 'deep', 'light')
        """
        if architecture == 'custom':
            return self._build_custom_model()
        elif architecture == 'deep':
            return self._build_deep_model()
        elif architecture == 'light':
            return self._build_light_model()
        else:
            return self._build_custom_model()

    def _build_custom_model(self):
        """Modelo CNN balanceado para clasificación multi-etiqueta"""
        model = Sequential([
            Input(shape=self.input_shape),

            # Primer bloque convolucional
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Segundo bloque convolucional
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Tercer bloque convolucional
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Cuarto bloque convolucional
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Capas de clasificación
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            # Capa de salida para multi-etiqueta (IMPORTANTE: sigmoid, no softmax)
            Dense(self.num_classes, activation='sigmoid')
        ])

        # Compilación para clasificación multi-etiqueta
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',  # ✅ Correcto para multi-label
            metrics=[
                'accuracy',
                BinaryAccuracy(name='binary_accuracy'),
                Precision(name='precision'),
                Recall(name='recall')
            ]
        )

        self.model = model
        return model

    def _build_deep_model(self):
        """Modelo CNN más profundo para mejor extracción de características"""
        model = Sequential([
            Input(shape=self.input_shape),

            # Bloque 1
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.2),

            # Bloque 2
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.2),

            # Bloque 3
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),

            # Bloque 4
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),

            # Bloque 5
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.4),

            # Clasificador
            GlobalAveragePooling2D(),
            Dense(1024, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                BinaryAccuracy(name='binary_accuracy'),
                Precision(name='precision'),
                Recall(name='recall')
            ]
        )

        self.model = model
        return model

    def _build_light_model(self):
        """Modelo CNN ligero para entrenamiento rápido"""
        model = Sequential([
            Input(shape=self.input_shape),

            # Bloque 1
            Conv2D(16, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.2),

            # Bloque 2
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.2),

            # Bloque 3
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),

            # Clasificador
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(self.num_classes, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                BinaryAccuracy(name='binary_accuracy'),
                Precision(name='precision'),
                Recall(name='recall')
            ]
        )

        self.model = model
        return model

    def get_callbacks(self, model_name='cnn_movie_model'):
        """
        Callbacks optimizados para clasificación multi-etiqueta
        """
        # Crear directorio si no existe
        os.makedirs('data/models', exist_ok=True)
        
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=10,  # Más paciencia para multi-label
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,  # Reducción más agresiva
                patience=5,
                verbose=1,
                min_lr=1e-7,
                cooldown=2
            ),
            ModelCheckpoint(
                filepath=f'data/models/{model_name}_best.h5',
                monitor='val_binary_accuracy',  # Usar binary_accuracy para multi-label
                save_best_only=True,
                verbose=1,
                save_weights_only=False
            )
        ]

    def train(self, train_generator, val_generator, epochs=100, callbacks=None):
        """
        Entrena el modelo con generadores de datos
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido construido. Llama a build_model() primero.")
        
        if callbacks is None:
            callbacks = self.get_callbacks()
        
        # Información del entrenamiento
        print(f"\n🔥 INICIANDO ENTRENAMIENTO CNN")
        print(f"📊 Arquitectura: {len(self.model.layers)} capas")
        print(f"🎯 Clases: {self.num_classes} géneros")
        print(f"📈 Épocas máximas: {epochs}")
        print(f"🖼️  Tamaño de entrada: {self.input_shape}")
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"✅ Entrenamiento completado!")
        return history

    def predict_genres(self, images, threshold=0.5):
        """
        Predice géneros para imágenes con umbral personalizable
        
        Args:
            images: Array de imágenes
            threshold: Umbral para considerar una clase como positiva
            
        Returns:
            predictions: Predicciones binarias
            probabilities: Probabilidades continuas
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido construido.")
        
        probabilities = self.model.predict(images)
        predictions = (probabilities > threshold).astype(int)
        
        return predictions, probabilities

    def get_model_summary(self):
        """Muestra resumen del modelo"""
        if self.model:
            print(f"\n📋 RESUMEN DEL MODELO CNN")
            print(f"🎯 Problema: Clasificación Multi-Etiqueta")
            print(f"🏷️  Clases: {self.num_classes} géneros")
            print(f"🖼️  Entrada: {self.input_shape}")
            print(f"⚙️  Función de pérdida: binary_crossentropy")
            print(f"📊 Activación final: sigmoid")
            print("-" * 50)
            return self.model.summary()
        else:
            print("❌ Modelo no construido aún. Llama a build_model() primero.")

    def print_model_info(self, model_name=None):
        """
        Imprime información detallada del modelo usando la función corregida
        """
        if self.model is None:
            print("❌ Modelo no construido aún. Llama a build_model() primero.")
            return
        
        if model_name is None:
            model_name = "CNN"
        
        print_model_info(self.model, model_name)

    def save_model(self, filepath):
        """Guarda el modelo entrenado"""
        if self.model is None:
            raise ValueError("El modelo no ha sido construido.")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        print(f"✅ Modelo guardado en: {filepath}")

    def load_model(self, filepath):
        """Carga un modelo pre-entrenado"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"✅ Modelo cargado desde: {filepath}")
        return self.model