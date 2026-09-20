#  We ensure proper path handling in Python
import Definitions
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.ModelController import ModelController

### Setup and configuration

st.set_page_config(
    layout="centered", page_title="ODS Text Classifier", page_icon="🌱"
)

### My vars

ctrl = ModelController()

### My UI starting here

with st.form(key="my_form"):
    uploaded_file = st.file_uploader(
        "Choose a xlsx file", accept_multiple_files=False, type="xlsx"
    )
    submit_button = st.form_submit_button(label="Submit")

if submit_button and uploaded_file is not None:
    input_df, is_valid = ctrl.load_input_data(uploaded_file)
    st.session_state["input_df"] = input_df if is_valid else None

input_df = st.session_state.get("input_df")

if input_df is not None:
    st.caption("✅ This is your data")
    event = st.dataframe(
        input_df,
        on_select="rerun",
        selection_mode="single-row",
        use_container_width=True,
    )
    st.caption("▶ Please select a row")

    if event is not None and event.selection.rows:        
        current_row_index = event.selection.rows[0]
        current_row = input_df.iloc[current_row_index]

        # Llama la clase de predicción procesando la fila seleccionada
        X_text, Y_real_raw, Y_pred_raw, Y_real_full, Y_pred_full = ctrl.predict(current_row)
        
        # Obtén el nombre de las clases
        class_names = ctrl.get_categories()

        col1, col2 = st.columns(2) 

        with col1:
            st.caption("🗣 Your Prediction")
            st.info(f"**Predicción:** {Y_pred_full}")
            with st.expander("Ver texto analizado"):
                st.write(X_text)

        with col2:
            st.caption("🎯 Your results")
            st.metric("Real", Y_real_full)
            st.metric("Prediction", Y_pred_full)
            
            if Y_real_raw != "N/A":
                match = Y_real_raw.strip() == Y_pred_raw.strip()
                st.markdown("✅ **¡Coincide!**" if match else "❌ **Difiere**")