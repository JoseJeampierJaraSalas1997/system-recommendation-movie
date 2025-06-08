import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

class MovieDataPreprocessor:
    def __init__(self, base_path, img_size=(224, 224)):
        self.base_path = base_path
        self.img_size = img_size
        self.label_columns = []
        self.label_binarizer = MultiLabelBinarizer()

    def load_data(self):
        """Carga imágenes y etiquetas desde carpeta Images y train.csv"""
        csv_path = os.path.join(self.base_path, 'train.csv')
        images_path = os.path.join(self.base_path, 'Images')

        df = pd.read_csv(csv_path)
        genre_labels = df['Genre'].str.split()  # Suponiendo que "Genre" tiene géneros separados por espacio

        self.label_columns = sorted(set(genre for genres in genre_labels for genre in genres))
        self.label_binarizer.fit([self.label_columns])  # Ajustar el binarizer

        X, y = [], []
        for _, row in df.iterrows():
            img_id = row['Id']
            img_file = os.path.join(images_path, f"{img_id}.jpg")
            if os.path.exists(img_file):
                try:
                    img = Image.open(img_file).convert('RGB')
                    img = img.resize(self.img_size)
                    img_array = np.array(img) / 255.0

                    X.append(img_array)

                    # Separar los géneros y codificarlos
                    genres = row['Genre'].split()
                    y.append(genres)
                except Exception as e:
                    print(f"Error al procesar {img_file}: {e}")

        # Binarizar todas las etiquetas
        y_binary = self.label_binarizer.transform(y)
        return np.array(X), np.array(y_binary)

    @property
    def label_encoder(self):
        return self.label_binarizer

# class MovieDataPreprocessor:
#     def __init__(self, base_path, img_size=(224, 224)):
#         self.base_path = base_path
#         self.img_size = img_size
#         self.label_columns = []

#     def load_data(self):
#         """Carga imágenes y etiquetas desde carpeta Images y train.csv"""
#         csv_path = os.path.join(self.base_path, 'train.csv')
#         images_path = os.path.join(self.base_path, 'Images')

#         df = pd.read_csv(csv_path)
#         self.label_columns = df.columns[2:]  # Ignora Id y Genre

#         X, y = [], []
#         for _, row in df.iterrows():
#             img_id = row['Id']
#             img_file = os.path.join(images_path, f"{img_id}.jpg")
#             if os.path.exists(img_file):
#                 try:
#                     img = Image.open(img_file).convert('RGB')
#                     img = img.resize(self.img_size)
#                     img_array = np.array(img) / 255.0

#                     X.append(img_array)
#                     y.append(row[self.label_columns].values.astype(np.uint8))
#                 except Exception as e:
#                     print(f"Error al procesar {img_file}: {e}")
#         return np.array(X), np.array(y)

    def create_data_splits(self, X, y, test_size=0.2, val_size=0.1):
        """Divide el dataset respetando multilabel (sin stratify por ahora)"""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_data_generators(self, X_train, y_train, X_val, y_val, batch_size=32):
        """Generadores de datos"""
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
        return train_generator, val_generator

    def visualize_samples(self, X, y, num_samples=12):
        """Visualiza muestras aleatorias con múltiples etiquetas"""
        fig, axes = plt.subplots(3, 4, figsize=(15, 12))
        axes = axes.ravel()
        indices = np.random.choice(len(X), num_samples, replace=False)

        for i, idx in enumerate(indices):
            axes[i].imshow(X[idx])
            labels = [label for label, val in zip(self.label_columns, y[idx]) if val == 1]
            axes[i].set_title("Géneros:\n" + "\n".join(labels), fontsize=8)
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig('results/plots/dataset_samples_multilabel.png', dpi=300)
        plt.show()

    def get_dataset_info(self, y):
        """Resumen del dataset"""
        class_totals = np.sum(y, axis=0)
        info = {
            'total_samples': len(y),
            'num_classes': len(self.label_columns),
            'class_distribution': dict(zip(self.label_columns, class_totals))
        }
        return info
