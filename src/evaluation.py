import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

class ModelEvaluator:
    def __init__(self, label_encoder):
        self.label_encoder = label_encoder
        
    def evaluate_model(self, model, X_test, y_test):
        """Evaluar modelo y calcular métricas"""
        # Predicciones
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Métricas
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Reporte de clasificación
        class_names = self.label_encoder.classes_
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        return metrics, report, y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path):
        """Crear matriz de confusión"""
        cm = confusion_matrix(y_true, y_pred)
        class_names = self.label_encoder.classes_
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Número de muestras'})
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.tight_layout()
        plt.savefig(f"{save_path}/{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_classification_report(self, report, model_name, save_path):
        """Visualizar reporte de clasificación"""
        # Convertir reporte a DataFrame
        df_report = pd.DataFrame(report).transpose()
        df_report = df_report.iloc[:-3, :-1]  # Remover filas de resumen
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_report, annot=True, cmap='RdYlBu_r', fmt='.3f',
                   cbar_kws={'label': 'Score'})
        plt.title(f'Reporte de Clasificación - {model_name}')
        plt.xlabel('Métricas')
        plt.ylabel('Clases')
        plt.tight_layout()
        plt.savefig(f"{save_path}/{model_name}_classification_report.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, results_dict, save_path):
        """Comparar múltiples modelos"""
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Crear DataFrame para comparación
        comparison_data = []
        for model_name in models:
            row = [model_name]
            for metric in metrics:
                row.append(results_dict[model_name]['metrics'][metric])
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data, 
                                   columns=['Model'] + [m.title() for m in metrics])
        
        # Gráfico de barras
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [results_dict[model][metric] for model in models]
            bars = ax.bar(models, values, color=['skyblue', 'lightcoral', 'lightgreen'][:len(models)])
            ax.set_title(f'Comparación - {metric.title()}')
            ax.set_ylabel(metric.title())
            ax.set_ylim(0, 1)
            
            # Agregar valores en las barras
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/models_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return df_comparison
    
    def save_metrics_to_csv(self, results_dict, save_path):
        """Guardar métricas en CSV"""
        all_results = []
        
        for model_name, results in results_dict.items():
            row = {'Model': model_name}
            row.update(results['metrics'])
            all_results.append(row)
        
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(f"{save_path}/model_comparison.csv", index=False)
        
        return df_results