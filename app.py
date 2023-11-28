import streamlit as st
import pandas as pd
import joblib

# Cargar modelos entrenados
#model_lr = joblib.load('modelo_regresion.pkl')
#model_svc = joblib.load('modelo_clasificacion.pkl')
model_svc = joblib.load('pred_svc.pkl')
model_lr = joblib.load('pred_lr.pkl')

# Crear la interfaz de Streamlit
st.title('Predicción de Match')

# Creando los inputs del usuario
age = st.number_input('Edad', min_value=18, max_value=100, value=30)
age_o = st.number_input('Edad de la otra persona', min_value=18, max_value=100, value=30)
gender = st.selectbox('Género', options=[0, 1], format_func=lambda x: 'Hombre' if x == 1 else 'Mujer')
race = st.selectbox('Raza', (1, 2, 3, 4, 5), format_func=lambda x: f'Raza {x}')
race_o = st.selectbox('Raza de la otra persona', (1, 2, 3, 4, 5), format_func=lambda x: f'Raza {x}')
field_cd = st.number_input('Código de campo de estudio', min_value=1, max_value=20, value=10)
career_c = st.number_input('Código de carrera', min_value=1, max_value=20, value=10)
imprace = st.slider('Importancia de la raza', 0, 10, 5)
imprelig = st.slider('Importancia de la religión', 0, 10, 5)
int_corr = st.slider('Coeficiente de correlación de intereses', -1.0, 1.0, 0.0)

# Botón para realizar predicciones
if st.button('Predecir Match'):
    # Preparar los datos para el modelo
    X_new = pd.DataFrame([[age, age_o, gender, race, race_o, field_cd, career_c, imprace, imprelig, int_corr]], 
                         columns=['age', 'age_o', 'gender', 'race', 'race_o', 'field_cd', 'career_c', 'imprace', 'imprelig', 'int_corr'])

    # Hacer las predicciones
    prediction_reg = model_lr.predict(X_new)[0]
    prediction_clf = model_svc.predict(X_new)[0]

    # Mostrar los resultados
    st.write(f'Predicción de match (Regresión): {prediction_reg:.2f}')
    st.write('Predicción de match (Clasificación):', 'Match' if prediction_clf == 1 else 'No Match')

