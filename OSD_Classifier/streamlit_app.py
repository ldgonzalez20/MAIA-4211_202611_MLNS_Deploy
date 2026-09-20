#  We ensure proper path handling in Python
import Definitions
import os
import numpy as np
import pandas as pd
import streamlit as st

from src.ModelController import ModelController

### Setup and configuration

st.set_page_config(
    layout="centered", page_title="Clasificador ODS", page_icon="🌱"
)

### My vars

ctrl = ModelController()

st.title("🌱 Clasificador de Textos ODS")
st.caption("📌 **Proyecto MAIA - MLNS (2026/09/20)**  |  Desarrollado por **Yuliana Delgado Osorio** y **Luz Dary González González**")

# Resumen del contexto y objetivo
with st.expander("ℹ️ Contexto y Objetivo del Proyecto (Agenda 2030 / UNFPA)"):
    st.markdown("""
    * **Contexto:** La ONU adoptó la *Agenda 2030* (17 ODS y 169 metas). Entidades como el *UNFPA* y organismos territoriales evalúan políticas públicas e impacto social mediante participación ciudadana.
    * **Desafío:** Interpretar y relacionar grandes volúmenes de información textual participativa con los ODS consume altos recursos y requiere validación de expertos.
    * **Objetivo:** Desarrollar una solución basada en *Natural Language Processing (NLP)* y *Machine Learning* para automatizar la clasificación semántica de textos hacia los 17 ODS, facilitando la toma de decisiones informadas.
    """)

# 🖼️ IMAGEN / PRINT SCREEN EN LA PANTALLA DE CARGA DE DATOS
img_path = os.path.join(Definitions.ROOT_DIR, "resources/images/presentacion.png")
if os.path.exists(img_path):
    st.image(
        img_path, 
        caption="Referencia / Descripción de los ODS", 
        width=500
    )
elif os.path.exists("presentacion.png"):
    st.image("presentacion.png", caption="Referencia / Descripción de los ODS", width=500)

st.divider()

# 📁 DESCRIPCIÓN DEL ARCHIVO EXCEL / DATASET A CARGAR
with st.expander("📂 Sobre el archivo de datos (`Datos_textosODS.xlsx`)"):
    st.markdown("""
    * **Fuente de origen:** Derivado del **OSDG Community Dataset (versión 2023)**, que contiene un total de 40.067 textos (incluyendo ~3.000 fuentes de la ONU, documentos públicos, resúmenes y reportes científicos/institucionales validados por una comunidad global de expertos y voluntarios).
    * **Preprocesamiento en español:** Los textos de referencia fueron traducidos al español usando **DeepL** y enriquecidos mediante técnicas de aumentación con la API de **ChatGPT**.
    * **Requisito del archivo:** El `.xlsx` debe contener las columnas `textos` y `ODS` para realizar la evaluación comparativa.
    """)

### My UI starting here

with st.form(key="my_form"):
    uploaded_file = st.file_uploader(
        "Elige un archivo Excel (.xlsx)",
        accept_multiple_files=False,
        type="xlsx",
    )
    submit_button = st.form_submit_button(label="Cargar archivo")

if submit_button and uploaded_file is not None:
    input_df, is_valid = ctrl.load_input_data(uploaded_file)
    st.session_state["input_df"] = input_df if is_valid else None

input_df = st.session_state.get("input_df")

if input_df is not None:
    st.caption("✅ Estos son tus datos")
    event = st.dataframe(
        input_df,
        on_select="rerun",
        selection_mode="single-row",
        use_container_width=True,
    )
    st.caption("▶ Por favor, selecciona una fila")

    if event is not None and event.selection.rows:
        current_row_index = event.selection.rows[0]
        current_row = input_df.iloc[current_row_index]

        # Llamar al método del controlador
        (
            X_text,
            Y_real_raw,
            Y_pred_raw,
            Y_real_full,
            Y_pred_full,
            probs,
            meta,
        ) = ctrl.get_prediction_details(current_row)

        st.divider()
        st.subheader("🎯 Evaluación de ODS")

        show_top3 = st.checkbox("Mostrar Top 3 predicciones", value=True)

        col1, col2 = st.columns(2)

        with col1:
            st.caption("🗣 Texto analizado")
            with st.expander("Ver contenido del texto", expanded=True):
                st.write(X_text)

        with col2:
            st.caption("🎯 Comparativa")
            # Reemplazado st.metric por markdown legible
            st.markdown(f"**Valor Real:**\n{Y_real_full}")
            if Y_real_raw != "N/A":
                match = Y_real_raw.strip() == Y_pred_raw.strip()
                st.markdown(
                    "✅ **¡Coincide!**" if match else "❌ **Difiere**"
                )

        st.divider()

        # Bloque de predicción con confianza y top 3
        pred_ods = Y_pred_raw
        if probs:
            classes = list(probs.keys())
            proba = np.array(list(probs.values()))

            st.success(
                f"### Predicción: ODS {pred_ods} — {ctrl.get_ods_description(pred_ods)}"
            )
            if pred_ods in list(classes):
                conf_idx = list(classes).index(pred_ods)
                confidence = proba[conf_idx]
                st.metric("Confianza del modelo", f"{confidence*100:.1f}%")

            if show_top3:
                order = np.argsort(proba)[::-1][:3]
                top3_df = pd.DataFrame(
                    {
                        "ODS": [classes[i] for i in order],
                        "Descripción": [
                            ctrl.get_ods_description(classes[i])
                            for i in order
                        ],
                        "Probabilidad": [
                            f"{proba[i]*100:.1f}%" for i in order
                        ],
                    }
                )
                st.table(top3_df)
        else:
            st.info(f"### Predicción: {Y_pred_full}")

        st.divider()
        st.caption(
            "Modelo: TF-IDF (min_df=3, max_df=0.85, 12000 términos) → TruncatedSVD (k=150) → "
            "LogisticRegression con búsqueda de hiperparámetros (GridSearchCV, F1-macro). "
            "Entrenado sobre el OSDG Community Dataset (subconjunto en español, 9656 textos, ODS 1-16)."
        )

        with st.expander("📈 Metadatos técnicos adicionales"):
            st.json(meta)