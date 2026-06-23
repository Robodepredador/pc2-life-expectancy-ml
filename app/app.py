# =============================================================================
# Expectativa de Vida Global — Análisis y Predicción
# Dataset WHO · 193 países · 2000–2015
# PC2 Agentes Inteligentes — USIL 2026-1
#
# Dashboard Streamlit de un solo archivo con dos paneles:
#   - Panel A: Análisis exploratorio interactivo  (informe sección 3.1)
#   - Panel B: Predicción con Random Forest         (informe sección 3.2)
# =============================================================================

import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# Configuración general de la página  (instrucción adicional 6)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Life Expectancy — PC2 USIL",
    page_icon="🌍",
    layout="wide",
)

# Rutas relativas a la raíz del repositorio (instrucción adicional 9).
# El app vive en app/app.py, por lo que subimos un nivel para llegar a la raíz.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_PATH = os.path.join(BASE_DIR, "data", "life_expectancy.csv")
URL_DATASET = (
    "https://raw.githubusercontent.com/"
    "Robodepredador/pc2-life-expectancy-ml/main/data/life_expectancy.csv"
)

TARGET = "Life expectancy"

# Columnas exactas del dataset (con espacios incluidos, instrucción adicional 8).
# Se usan tras normalizar los nombres del CSV con .str.strip().
DATASET_COLUMNS = [
    "Country", "Year", "Status", "Life expectancy", "Adult Mortality",
    "infant deaths", "Alcohol", "percentage expenditure", "Hepatitis B",
    "Measles", "BMI", "under-five deaths", "Polio", "Total expenditure",
    "Diphtheria", "HIV/AIDS", "GDP", "Population",
    "thinness  1-19 years", "thinness 5-9 years",
    "Income composition of resources", "Schooling",
]


# =============================================================================
# Carga de modelos y datos  (informe sección 2 — preparación)
# =============================================================================
@st.cache_resource(show_spinner="Cargando modelos entrenados...")
def load_models():
    """Carga el modelo, el scaler, el label encoder y el metadata (.pkl).

    Usa @st.cache_resource para no recargar los objetos en cada interacción
    (instrucción adicional 2). Devuelve None en el archivo que falte para que
    el llamador muestre un error descriptivo (instrucción adicional 5).
    """
    archivos = {
        "model": "mejor_modelo_rf.pkl",
        "scaler": "scaler.pkl",
        "le": "label_encoder.pkl",
        "metadata": "metadata.pkl",
    }
    objetos = {}
    faltantes = []
    for clave, nombre in archivos.items():
        ruta = os.path.join(MODEL_DIR, nombre)
        if not os.path.exists(ruta):
            faltantes.append(f"model/{nombre}")
            objetos[clave] = None
            continue
        objetos[clave] = joblib.load(ruta)
    return objetos, faltantes


@st.cache_data(show_spinner="Cargando dataset...")
def load_data_from_source(source):
    """Carga el CSV desde una ruta/URL/archivo subido y normaliza columnas.

    El dataset original trae espacios sobrantes en los nombres de columna
    (p. ej. 'Life expectancy ', ' BMI '); se limpian con .str.strip() para
    que coincidan con metadata['features'] y con TARGET. Usa @st.cache_data
    (instrucción adicional 2).
    """
    df = pd.read_csv(source)
    df.columns = df.columns.str.strip()
    return df


def get_dataframe():
    """Estrategia de carga del dataset (informe sección 2).

    1) Intenta el CSV local del repositorio (model deploy en Streamlit Cloud).
    2) Si no existe, intenta la URL raw de GitHub.
    3) Como último recurso ofrece un st.file_uploader (fallback manual).
    """
    # 1) CSV local versionado en el repo
    if os.path.exists(DATA_PATH):
        try:
            return load_data_from_source(DATA_PATH)
        except Exception:
            pass
    # 2) URL raw de GitHub
    try:
        return load_data_from_source(URL_DATASET)
    except Exception:
        st.warning(
            "No se pudo cargar el dataset automáticamente desde el repositorio "
            "ni desde la URL de GitHub. Sube el CSV manualmente para continuar."
        )
    # 3) Fallback: subida manual
    archivo = st.file_uploader("Sube life_expectancy.csv", type=["csv"])
    if archivo is not None:
        return load_data_from_source(archivo)
    return None


# -----------------------------------------------------------------------------
# Inicialización: cargar modelos y validar
# -----------------------------------------------------------------------------
objetos, faltantes = load_models()
if faltantes:
    st.error(
        "No se encontraron los siguientes archivos de modelo necesarios:\n\n"
        + "\n".join(f"- `{f}`" for f in faltantes)
        + "\n\nVerifica que la carpeta `model/` esté en la raíz del repositorio."
    )
    st.stop()

model = objetos["model"]
scaler = objetos["scaler"]
le = objetos["le"]
metadata = objetos["metadata"]

FEATURES = metadata["features"]          # orden EXACTO para predecir (instr. 4)
FEATURE_MIN = metadata["feature_min"]
FEATURE_MAX = metadata["feature_max"]
FEATURE_MEDIAN = metadata["feature_median"]

df = get_dataframe()
if df is None:
    st.info("Esperando el dataset para mostrar el dashboard...")
    st.stop()


def encode_status(valor_texto):
    """Convierte 'Developed'/'Developing' a su código numérico.

    Intenta usar el LabelEncoder entrenado; si falla, aplica el mapeo
    documentado en el plan (Developed=0, Developing=1).
    """
    try:
        return int(le.transform([valor_texto])[0])
    except Exception:
        return 0 if valor_texto == "Developed" else 1


# =============================================================================
# Encabezado general
# =============================================================================
st.title("🌍 Expectativa de Vida Global — Análisis y Predicción")
st.caption("Dataset WHO · 193 países · 2000–2015 · Modelo: Random Forest Regressor")

panel = st.sidebar.radio(
    "Navegación",
    ["📊 Panel A — Análisis de datos", "🔮 Panel B — Predicción"],
)
st.sidebar.markdown("---")


# =============================================================================
# PANEL A — ANÁLISIS DE DATOS  (informe sección 3.1 · rúbrica 1.5 pts)
# =============================================================================
def render_panel_a():
    st.header("📊 Panel A — Análisis exploratorio interactivo")

    # ---- Filtros en el sidebar (informe sección 3.1) -----------------------
    st.sidebar.subheader("Filtros")

    anios = sorted(df["Year"].dropna().unique().astype(int))
    anio_sel = st.sidebar.selectbox("Año", ["Todos"] + anios, index=0)

    status_opts = sorted(df["Status"].dropna().unique())
    status_sel = st.sidebar.multiselect(
        "Status", status_opts, default=status_opts
    )

    le_min = float(np.nanmin(df[TARGET]))
    le_max = float(np.nanmax(df[TARGET]))
    rango_le = st.sidebar.slider(
        "Rango de expectativa de vida (años)",
        min_value=round(le_min, 1),
        max_value=round(le_max, 1),
        value=(round(le_min, 1), round(le_max, 1)),
    )

    # ---- Aplicar filtros ---------------------------------------------------
    dff = df.copy()
    if anio_sel != "Todos":
        dff = dff[dff["Year"] == anio_sel]
    if status_sel:
        dff = dff[dff["Status"].isin(status_sel)]
    dff = dff[
        (dff[TARGET] >= rango_le[0]) & (dff[TARGET] <= rango_le[1])
    ]
    dff = dff.dropna(subset=[TARGET])

    if dff.empty:
        st.warning("No hay registros que cumplan los filtros seleccionados.")
        return

    # ---- KPIs (st.metric) --------------------------------------------------
    pais_max = dff.loc[dff[TARGET].idxmax()]
    pais_min = dff.loc[dff[TARGET].idxmin()]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registros activos", f"{len(dff):,}")
    k2.metric("Expectativa promedio", f"{dff[TARGET].mean():.1f} años")
    k3.metric(
        "Mayor expectativa",
        f"{pais_max[TARGET]:.1f} años",
        help=f"{pais_max['Country']} ({int(pais_max['Year'])})",
    )
    k4.metric(
        "Menor expectativa",
        f"{pais_min[TARGET]:.1f} años",
        help=f"{pais_min['Country']} ({int(pais_min['Year'])})",
    )
    st.markdown(
        f"🥇 **Mayor:** {pais_max['Country']} ({int(pais_max['Year'])}) — "
        f"{pais_max[TARGET]:.1f} años  ·  "
        f"🥉 **Menor:** {pais_min['Country']} ({int(pais_min['Year'])}) — "
        f"{pais_min[TARGET]:.1f} años"
    )
    st.markdown("---")

    # ---- Gráfico A1 — Scatter GDP vs Expectativa de vida -------------------
    st.subheader("A1 · GDP per cápita vs Expectativa de vida")
    dff_scatter = dff.dropna(subset=["GDP", TARGET, "Schooling"])
    fig_a1 = px.scatter(
        dff_scatter,
        x="GDP",
        y=TARGET,
        color="Status",
        size="Schooling",
        log_x=True,
        hover_data=["Country", "Year", TARGET, "GDP"],
        labels={"GDP": "GDP per cápita (escala log)", TARGET: "Expectativa (años)"},
        title="Relación entre riqueza, escolaridad y longevidad",
    )
    st.plotly_chart(fig_a1, use_container_width=True)

    # ---- Gráfico A2 — Mapa de calor de correlación ------------------------
    st.subheader("A2 · Matriz de correlación de Pearson")
    num_cols = dff.select_dtypes(include=[np.number]).drop(columns=["Year"], errors="ignore")
    corr = num_cols.corr(numeric_only=True).round(2)
    fig_a2 = px.imshow(
        corr,
        color_continuous_scale="RdYlGn",
        zmin=-1,
        zmax=1,
        text_auto=True,
        aspect="auto",
        title="Correlación entre variables numéricas",
    )
    fig_a2.update_layout(height=700)
    st.plotly_chart(fig_a2, use_container_width=True)

    # ---- Gráfico A3 — Boxplot por Status ----------------------------------
    st.subheader("A3 · Distribución de la expectativa por Status")
    fig_a3 = px.box(
        dff,
        x="Status",
        y=TARGET,
        color="Status",
        points="outliers",
        labels={TARGET: "Expectativa de vida (años)"},
        title="Dispersión y outliers según grado de desarrollo",
    )
    st.plotly_chart(fig_a3, use_container_width=True)

    # ---- Gráfico A4 — Serie temporal --------------------------------------
    st.subheader("A4 · Evolución temporal por Status")
    serie = (
        dff.groupby(["Year", "Status"])[TARGET]
        .mean()
        .reset_index()
        .sort_values("Year")
    )
    fig_a4 = px.line(
        serie,
        x="Year",
        y=TARGET,
        color="Status",
        markers=True,
        labels={TARGET: "Expectativa promedio (años)", "Year": "Año"},
        title="Promedio anual de expectativa de vida (2000–2015)",
    )
    st.plotly_chart(fig_a4, use_container_width=True)


# =============================================================================
# PANEL B — PREDICCIÓN  (informe sección 3.2 · rúbrica 1.5 pts)
# =============================================================================
def slider_feature(label, feature, fmt="%.2f"):
    """Crea un slider para una feature usando min/median/max del metadata.

    (instrucción adicional 3). Devuelve el valor seleccionado por el usuario.
    """
    vmin = float(FEATURE_MIN[feature])
    vmax = float(FEATURE_MAX[feature])
    vmed = float(FEATURE_MEDIAN[feature])
    # Salvaguarda por si median quedara fuera de [min, max]
    vmed = min(max(vmed, vmin), vmax)
    if vmin == vmax:
        vmax = vmin + 1.0
    return st.slider(label, min_value=vmin, max_value=vmax, value=vmed, format=fmt)


def render_panel_b():
    st.header("🔮 Panel B — Predicción de expectativa de vida")
    st.write(
        "Ingresa los indicadores de un país hipotético y el modelo "
        "**Random Forest** estimará su expectativa de vida."
    )

    # Promedios de referencia para contextualizar el resultado
    prom_global = float(df[TARGET].mean())
    prom_status = df.groupby("Status")[TARGET].mean().to_dict()

    # ---- Entradas del usuario en 3 columnas (informe sección 3.2) ---------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🩺 Mortalidad**")
        adult_mortality = slider_feature("Adult Mortality (por 1000 adultos)", "Adult Mortality")
        infant_deaths = slider_feature("Infant deaths (por 1000 nacimientos)", "infant deaths")
        under_five = slider_feature("Under-five deaths (por 1000 nacimientos)", "under-five deaths")

    with col2:
        st.markdown("**💰 Salud y economía**")
        gdp = slider_feature("GDP (USD per cápita)", "GDP")
        total_exp = slider_feature("Total expenditure (% gasto en salud)", "Total expenditure")
        hiv = slider_feature("HIV/AIDS (muertes por 1000 hab.)", "HIV/AIDS")

    with col3:
        st.markdown("**🎓 Desarrollo**")
        schooling = slider_feature("Schooling (años de escolaridad)", "Schooling")
        icor = slider_feature("Income composition of resources (0-1)", "Income composition of resources")
        status_txt = st.selectbox("Status", ["Developing", "Developed"])

    # Valores ingresados explícitamente por el usuario
    user_values = {
        "Adult Mortality": adult_mortality,
        "infant deaths": infant_deaths,
        "under-five deaths": under_five,
        "GDP": gdp,
        "Total expenditure": total_exp,
        "HIV/AIDS": hiv,
        "Schooling": schooling,
        "Income composition of resources": icor,
        "Status": encode_status(status_txt),
    }

    st.markdown("---")

    # ---- Botón de predicción (instrucción adicional, sección 3.2) ---------
    if st.button("🔮 Predecir expectativa de vida", type="primary"):
        with st.spinner("Calculando predicción..."):
            # 1) Construir el vector en el orden EXACTO de metadata['features']
            #    (instrucción adicional 4). Las features no expuestas usan la
            #    mediana del dataset (instrucción adicional 3 / sección 3.2).
            fila = []
            fuera_de_rango = []
            for feat in FEATURES:
                if feat in user_values:
                    val = float(user_values[feat])
                    # Verificar rango observado (instrucción adicional, warning)
                    if feat != "Status" and (
                        val < float(FEATURE_MIN[feat]) or val > float(FEATURE_MAX[feat])
                    ):
                        fuera_de_rango.append(feat)
                else:
                    val = float(FEATURE_MEDIAN[feat])
                fila.append(val)

            X = np.array(fila, dtype=float).reshape(1, -1)

            # 2) Escalar  → 3) Predecir
            X_scaled = scaler.transform(X)
            pred = float(model.predict(X_scaled)[0])

        # ---- 4) Mostrar el resultado --------------------------------------
        if fuera_de_rango:
            st.warning(
                "Algunos valores están fuera del rango observado en el dataset, "
                "la predicción puede ser menos confiable: "
                + ", ".join(f"`{f}`" for f in fuera_de_rango)
            )

        st.success(f"### Expectativa de vida estimada: **{pred:.1f} años**")

        # Comparaciones de contexto
        diff_global = pred - prom_global
        flecha_g = "↑" if diff_global >= 0 else "↓"
        prom_grp = prom_status.get(status_txt, prom_global)
        diff_grp = pred - prom_grp
        flecha_grp = "↑" if diff_grp >= 0 else "↓"

        c1, c2 = st.columns(2)
        c1.metric(
            "vs promedio mundial",
            f"{pred:.1f} años",
            f"{flecha_g} {abs(diff_global):.1f} años (global {prom_global:.1f})",
        )
        c2.metric(
            f"vs promedio {status_txt}",
            f"{pred:.1f} años",
            f"{flecha_grp} {abs(diff_grp):.1f} años (grupo {prom_grp:.1f})",
        )

        # Nivel de desarrollo humano según ICOR
        if icor >= 0.7:
            nivel = "alto"
        elif icor >= 0.5:
            nivel = "medio"
        else:
            nivel = "bajo"

        st.info(
            f"Un país con estas características tendría una expectativa de vida "
            f"estimada de **{pred:.1f} años**. Esto lo ubicaría en el grupo de "
            f"países **{status_txt}** con **{nivel}** desarrollo humano "
            f"(ICOR = {icor:.2f})."
        )

    with st.expander("ℹ️ ¿Cómo funciona la predicción?"):
        st.markdown(
            "- Se construye un vector con las **19 features** en el orden exacto "
            "esperado por el modelo.\n"
            "- Las variables no editables toman la **mediana** del dataset.\n"
            "- El vector se **estandariza** con el `StandardScaler` entrenado y "
            "se pasa al **Random Forest Regressor**."
        )


# =============================================================================
# Enrutado de paneles
# =============================================================================
if panel.startswith("📊"):
    render_panel_a()
else:
    render_panel_b()

st.sidebar.markdown("---")
st.sidebar.caption("PC2 Agentes Inteligentes · USIL 2026-1")
