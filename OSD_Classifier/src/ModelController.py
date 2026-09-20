import Definitions
import numpy as np
import os.path as osp
import pandas as pd
import io
import joblib

from src.DataPreprocessing import DataPreprocessing

class ModelController:
    # Diccionario oficial de ODS
    ODS_DICC = {
        "1": "Fin de la pobreza", "2": "Cero Hambre", "3": "Salud y bienestar",
        "4": "Educación de calidad", "5": "Igualdad de género", "6": "Agua limpia y saneamiento",
        "7": "Energía asequible y no contaminante", "8": "Trabajo decente y crecimiento económico",
        "9": "Industria, innovación e infraestructura", "10": "Reducción de las desigualdades",
        "11": "Ciudades y comunidades sostenibles", "12": "Producción y consumo responsables",
        "13": "Acción por el clima", "14": "Vida submarina", "15": "Vida de ecosistemas terrestres",
        "16": "Paz, justicia e instituciones sólidas", "17": "Alianzas para lograr los objetivos"
    }

    def __init__(self):
        print("ModelController.__init__ ->")
        self.model_path = osp.join(Definitions.ROOT_DIR, "resources/models")
        
        # Rutas de los artefactos del pipeline de texto
        self.tfidf_path = osp.join(self.model_path, "tfidf_vectorizer.joblib")
        self.svd_path = osp.join(self.model_path, "svd_model.joblib")
        self.classifier_path = osp.join(self.model_path, "ods_classifier.joblib")

        # Cargar los modelos
        self.tfidf = joblib.load(self.tfidf_path)
        self.svd = joblib.load(self.svd_path)
        self.classifier = joblib.load(self.classifier_path)

        self.input_df = pd.DataFrame()
        self.d_processing = DataPreprocessing()

    def get_ods_description(self, code):
        code_str = str(code).replace('.0', '').strip()
        name = self.ODS_DICC.get(code_str, "Desconocido")
        return f"ODS {code_str}: {name}" if name != "Desconocido" else f"ODS {code_str}"

    def validate_data(self, df):
        return self.d_processing.get_columns().issubset(set(df.columns))
    
    def get_categories(self):
        print("ModelController.get_categories ->")
        return [str(i) for i in range(1, 18)]

    def load_input_data(self, uploaded_file):
        print("ModelController.load_input_data ->")
        try:
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
            self.input_df = pd.read_excel(io.BytesIO(uploaded_file.read()))
            is_valid = self.validate_data(self.input_df)
            return self.input_df, is_valid
        except Exception as e:
            print(f"Error detallado al leer Excel: {e}")
            raise ValueError(f"Ocurrió un error al leer la información de entrada: {e}")

    def predict(self, current_row):
        text_content = str(current_row['textos'])
        Y_real_raw = str(current_row['ODS']).replace('.0', '').strip() if 'ODS' in current_row else "N/A"
        
        X_tfidf = self.tfidf.transform([text_content])
        X_reduced = self.svd.transform(X_tfidf)
        y_pred_raw = str(self.classifier.predict(X_reduced)[0]).replace('.0', '').strip()
        
        Y_real_full = self.get_ods_description(Y_real_raw) if Y_real_raw != "N/A" else "N/A"
        y_pred_full = self.get_ods_description(y_pred_raw)
        
        return text_content, Y_real_raw, y_pred_raw, Y_real_full, y_pred_full

    def get_prediction_details(self, current_row):
        print("ModelController.get_prediction_details ->")
        text_content, Y_real_raw, y_pred_raw, Y_real_full, y_pred_full = self.predict(current_row)
        
        X_tfidf = self.tfidf.transform([text_content])
        X_reduced = self.svd.transform(X_tfidf)
        
        probs = None
        if hasattr(self.classifier, "predict_proba"):
            probs_array = self.classifier.predict_proba(X_reduced)[0]
            classes = [str(c).replace('.0', '').strip() for c in self.classifier.classes_]
            probs = {str(c): float(p) for c, p in zip(classes, probs_array)}
            
        meta = {
            "tfidf_features": getattr(self.tfidf, "max_features", "N/A"),
            "svd_components": getattr(self.svd, "n_components", "N/A"),
            "explained_variance_ratio": float(np.sum(self.svd.explained_variance_ratio_)) if hasattr(self.svd, "explained_variance_ratio_") else "N/A"
        }
        
        return text_content, Y_real_raw, y_pred_raw, Y_real_full, y_pred_full, probs, meta