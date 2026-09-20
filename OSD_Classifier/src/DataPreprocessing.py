import numpy as np
import pandas as pd

class DataPreprocessing:

    def __init__(self):
        print("DataPreprocessing.__init__ ->")
        self.expected_columns = {'textos', 'ODS'}
        self.ODS_DICC = {
            "1": "Fin de la pobreza", "2": "Cero Hambre", "3": "Salud y bienestar",
            "4": "Educación de calidad", "5": "Igualdad de género", "6": "Agua limpia y saneamiento",
            "7": "Energía asequible y no contaminante", "8": "Trabajo decente y crecimiento económico",
            "9": "Industria, innovación e infraestructura", "10": "Reducción de las desigualdades",
            "11": "Ciudades y comunidades sostenibles", "12": "Producción y consumo responsables",
            "13": "Acción por el clima", "14": "Vida submarina", "15": "Vida de ecosistemas terrestres",
            "16": "Paz, justicia e instituciones sólidas", "17": "Alianzas para lograr los objetivos"
        }

    def transform(self, df):
        print("DataPreprocessing.transform ->")
        if df is None:
            return pd.DataFrame()
        df_clean = df.copy()
        # Asegurar que la columna de textos sea string y manejar nulos
        if 'textos' in df_clean.columns:
            df_clean['textos'] = df_clean['textos'].fillna('').astype(str)
        return df_clean

    def get_columns(self):
        print("DataPreprocessing.get_columns ->")
        res = ['textos', 'ODS']
        return set(res)

    def get_categories(self):
        return [str(i) for i in range(1, 18)]

    def get_cat_name(self, index_or_code):
        print("DataPreprocessing.get_cat_name ->")
        code_str = str(index_or_code).strip()
        return self.ODS_DICC.get(code_str, "Desconocido")