
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, chi2_contingency

st.set_page_config(page_title='EDA Infractores vs Control', layout='wide')
st.title('Análisis Exploratorio de Datos - Infractores vs Control')

@st.cache_data
def cargar_datos():
    return pd.read_excel("datos.xlsx")

df = cargar_datos()

# Variables correctas para cuantitativas
variables_cuantitativas = [
    'vs_edad', 'vs_cantidadhijos', 'vs_cantidadhermanos',
    'vs_numero_ocupantes_vivienda', 'pc_ineco', 'pc_moca'
]

# Variables correctas para categóricas
variables_categoricas = [
    'vs_tienehijos', 'vs_nivel_educacion', 'vs_desercion_escolar',
    'vs_pandillismo', 'vs_consumospa', 'vs_tipo_familia',
    'vs_violencia_intrafamiliar', 'vs_tiene_televisor',
    'vs_tiene_celular', 'vs_tiene_computador', 'vs_tiene_internet',
    'vs_ingreso_familiar', 'vs_tenencia_vivienda', 'vs_estrato_socioeconómico'
]

seccion = st.sidebar.radio("Ir a:", ["1. Vista general", "2. Variables cuantitativas", "3. Variables categóricas"])

if seccion == "1. Vista general":
    st.subheader("Vista general del conjunto de datos")
    st.dataframe(df)
    st.write("Número de filas y columnas:", df.shape)

elif seccion == "2. Variables cuantitativas":
    st.subheader("Estadísticos descriptivos por grupo")

    resumen = []
    for var in variables_cuantitativas:
        for grupo in df['sujeto'].unique():
            datos = df[df['sujeto'] == grupo][var]
            fila = {
                'Variable': var,
                'Grupo': grupo,
                'Media ± DE': f"{datos.mean():.2f} ± {datos.std():.2f}",
                'Mediana [RI]': f"{datos.median():.2f} [{(datos.quantile(0.75)-datos.quantile(0.25)):.2f}]",
                'Mín – Máx': f"{datos.min()} – {datos.max()}",
                'Moda': datos.mode().iloc[0] if not datos.mode().empty else '—'
            }
            resumen.append(fila)
    st.dataframe(pd.DataFrame(resumen))

    st.subheader("Resultados de prueba Mann-Whitney por variable")
    resultados = []
    for var in variables_cuantitativas:
        grupo1 = df[df['sujeto'] == 'Infractor'][var]
        grupo2 = df[df['sujeto'] == 'control'][var]
        u, p = mannwhitneyu(grupo1, grupo2)
        resultados.append({
            'Variable': var,
            'U': u,
            'p-valor': p,
            'p < 0.05': 'Sí' if p < 0.05 else 'No'
        })
    st.dataframe(pd.DataFrame(resultados))

    st.subheader("Boxplot interactivo por variable")
    var_select = st.selectbox("Selecciona variable cuantitativa", variables_cuantitativas)
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='sujeto', y=var_select, palette='pastel', ax=ax)
    ax.set_title(f'Distribución de {var_select}')
    st.pyplot(fig)

    st.subheader("Interpretación")
    st.markdown("Se observaron diferencias en edad, cantidad de hermanos, número de ocupantes y puntajes cognitivos entre infractores y controles, confirmadas por la prueba de Mann-Whitney en la mayoría de las variables.")

elif seccion == "3. Variables categóricas":
    st.subheader("Distribución por variable")
    var_cat = st.selectbox("Selecciona variable categórica", variables_categoricas)
    tabla = pd.crosstab(df[var_cat], df['sujeto'], normalize='columns') * 100
    fig, ax = plt.subplots()
    tabla.plot(kind='bar', ax=ax)
    ax.set_title(f'Distribución de {var_cat}')
    ax.set_ylabel('Porcentaje (%)')
    st.pyplot(fig)

    st.subheader("Resultados de prueba Chi-cuadrado por variable")
    resultados_cat = []
    for var in variables_categoricas:
        tabla = pd.crosstab(df[var], df['sujeto'])
        if tabla.shape[0] > 1 and tabla.shape[1] > 1:
            chi2, p, _, _ = chi2_contingency(tabla)
            resultados_cat.append({
                'Variable': var,
                'Chi²': round(chi2, 3),
                'p-valor': round(p, 4),
                'p < 0.05': 'Sí' if p < 0.05 else 'No'
            })
    st.dataframe(pd.DataFrame(resultados_cat))

    st.subheader("Interpretación")
    st.markdown("Las variables categóricas muestran grandes diferencias entre grupos. Los infractores tienen mayor exposición a condiciones de riesgo, menor acceso a bienes y predominio en estratos bajos.")
