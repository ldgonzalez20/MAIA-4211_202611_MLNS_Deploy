import io
import pandas as pd
import os.path as osp
import Definitions
import joblib
from src.DataPreprocessing import DataPreprocessing



class ModelController:
   # Diccionario de ODS
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
        # Ruta base de los modelos
        self.model_base_path = osp.join(Definitions.ROOT_DIR, "resources/models")
        
        # Rutas de los artefactos de texto (TF-IDF, SVD y Clasificador ODS)
        self.tfidf_path = osp.join(self.model_base_path, "tfidf_vectorizer.joblib")
        self.svd_path = osp.join(self.model_base_path, "svd_model.joblib")
        self.classifier_path = osp.join(self.model_base_path, "ods_classifier.joblib")

        # Cargar los modelos
        self.tfidf = joblib.load(self.tfidf_path)
        self.svd = joblib.load(self.svd_path)
        self.classifier = joblib.load(self.classifier_path)

        # Inicializar variables
        self.input_df = pd.DataFrame()
        # Clase de preprocesamiento de la información 
        self.d_processing = DataPreprocessing()

    def validate_data(self, df):
        # Valida que existan las columnas esperadas en el Excel de ODS
        expected_cols = {'textos', 'ODS'}
        return expected_cols.issubset(set(df.columns))
    
    def get_categories(self):
        print("ModelController.get_categories ->")
        # Clases u ODS disponibles (ej. 1 a 17 o las etiquetas únicas de tus datos)
        return [str(i) for i in range(1, 18)]

    def load_input_data(self, uploaded_file):
        print("ModelController.load_input_data ->")
        try:
            # Asegurar que el puntero del archivo esté al inicio
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
                
            # Leer el archivo Excel de manera segura
            self.input_df = pd.read_excel(io.BytesIO(uploaded_file.read()))
            is_valid = self.validate_data(self.input_df)
            return self.input_df, is_valid

        except Exception as e:
            print(f"Error detallado al leer Excel: {e}")
            raise ValueError(f"Ocurrió un error al leer la información de entrada: {e}")

    # Traer la descripción del ODS
    def get_ods_description(self, code):
        code_str = str(code).strip()
        name = self.ODS_DICC.get(code_str, "Desconocido")
        return f"ODS {code_str}: {name}" if name != "Desconocido" else f"ODS {code_str}"    

    def predict(self, current_row):
        print("ModelController.predict ->")
        text_content = str(current_row['textos'])
        Y_real_raw = str(current_row['ODS']) if 'ODS' in current_row else "N/A"
        
        # Inferencia
        X_tfidf = self.tfidf.transform([text_content])
        X_reduced = self.svd.transform(X_tfidf)
        y_pred_raw = str(self.classifier.predict(X_reduced)[0])
        
        # Formatos con nombres completos
        Y_real_full = self.get_ods_description(Y_real_raw) if Y_real_raw != "N/A" else "N/A"
        y_pred_full = self.get_ods_description(y_pred_raw)
        
        return text_content, Y_real_raw, y_pred_raw, Y_real_full, y_pred_full
