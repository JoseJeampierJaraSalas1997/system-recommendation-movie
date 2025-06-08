import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

class MLPModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=25):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self, hidden_layers=[1024, 512, 256], dropout_rates=[0.5, 0.4, 0.3]):
        """Construir modelo MLP"""
        model = Sequential()
        
        # Aplanar entrada
        model.add(Flatten(input_shape=self.input_shape))
        
        # Capas ocultas
        for i, (units, dropout) in enumerate(zip(hidden_layers, dropout_rates)):
            model.add(Dense(units, activation='relu', name=f'dense_{i+1}'))
            model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
            model.add(Dropout(dropout, name=f'dropout_{i+1}'))
        
        model.add(Dense(self.num_classes, activation='sigmoid', name='output'))
        
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

    def get_callbacks(self, model_name):
        """Configurar callbacks para entrenamiento"""
        os.makedirs('data/models', exist_ok=True)
        
        return [
            EarlyStopping(
                monitor='val_binary_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_binary_accuracy',
                factor=0.3,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'data/models/{model_name}_best.h5',
                monitor='val_binary_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
    
    def train(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=32, model_name='mlp_model'):
        """Entrenar modelo"""
        callbacks = self.get_callbacks(model_name)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

    def load_trained_model(self, model_path):
        """Cargar modelo previamente entrenado"""
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"Modelo cargado desde: {model_path}")
        else:
            print(f"Ruta no válida: {model_path}")

    def get_model_summary(self):
        """Obtener resumen del modelo"""
        if self.model:
            return self.model.summary()
        else:
            print("Modelo no construido aún.")