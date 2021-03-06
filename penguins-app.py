import streamlit as st
import pandas as pd
import numpy as np
import pickle
# from sklearn.ensemble import RandomForestClassifier

st.write("""
# Aplicación para predicir Pigüinos

Esta aplicación predice tres especies diferentes **Adelie, Chinstrap y Gentoo**!

Los datos fueron obtenidos de [Estación Palmer Antártida LTER](https://pallter.marine.rutgers.edu/).

Miembro de la Red de Investigación Ecológica a Largo Plazo.
""")

st.sidebar.header('Funciones de entrada del usuario')

st.sidebar.markdown("""
[Ejemplo de archivo de entrada CSV](https://raw.githubusercontent.com/MelDusan/CSV/main/penguins_cleaned.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Carga tu archivo CSV de entrada", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Islas', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sexo', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Longitud de pico (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Profundidad de pico (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Longitud de la aleta (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Masa corporal (g)', 2700.0, 6300.0, 4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('Funciones de entrada de usuario')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('A la espera de que se cargue el archivo CSV. Actualmente usando parámetros de entrada de ejemplo (que se muestran a continuación).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Predicción')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Probabilidad de Predicción')
st.write(prediction_proba)