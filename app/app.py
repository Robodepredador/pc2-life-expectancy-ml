# =============================================================================
# Expectativa de Vida Global — Análisis y Predicción
# Dataset WHO · 193 países · 2000–2015
# PC2 Agentes Inteligentes — USIL 2026-1
#
# Dashboard Streamlit de un solo archivo con dos paneles:
#   - Panel A: Análisis exploratorio interactivo  (informe sección 3.1)
#   - Panel B: Predicción con Random Forest         (informe sección 3.2)
#
# Mejoras de Interacción Humano-Computadora (IHC) aplicadas:
#   - Jerarquía visual y agrupación Gestalt (tarjetas + pestañas)
#   - Consistencia (paleta de color única para Status en todo el dashboard)
#   - Reconocer en vez de recordar (textos guía bajo cada gráfico)
#   - Visibilidad del estado del sistema (resumen de filtros activos, spinners)
#   - Feedback claro (formulario de predicción + medidor visual del resultado)
#   - Diseño minimalista (CSS limpio, espaciado consistente)
# =============================================================================

import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Configuración general de la página  (instrucción adicional 6)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Life Expectancy — PC2 USIL",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta consistente para Status (IHC: consistencia y estándares).
COLOR_DEVELOPED = "#0E7C7B"     # verde azulado
COLOR_DEVELOPING = "#E1A140"    # ámbar
COLOR_MAP = {"Developed": COLOR_DEVELOPED, "Developing": COLOR_DEVELOPING}

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


# -----------------------------------------------------------------------------
# Estilos (CSS) — IHC: estética minimalista y jerarquía visual
# -----------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        /* Contenedor principal más respirado */
        .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }

        /* Tarjetas de KPI a partir de st.metric (agrupación Gestalt) */
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E3EAEC;
            border-left: 5px solid #0E7C7B;
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(16,40,48,0.06);
        }
        div[data-testid="stMetricLabel"] p {
            font-size: 0.82rem; color: #5B6B71; font-weight: 600;
            text-transform: uppercase; letter-spacing: .03em;
        }
        div[data-testid="stMetricValue"] { font-weight: 700; }

        /* Encabezado de sección con barra de acento */
        .sec-head {
            border-left: 6px solid #0E7C7B; padding: 2px 0 2px 12px;
            margin: 8px 0 2px 0;
        }
        .sec-head h3 { margin: 0; font-size: 1.25rem; color: #1A2B33; }
        .sec-head p { margin: 2px 0 0 0; color: #5B6B71; font-size: 0.9rem; }

        /* Tarjeta de resultado de predicción */
        .pred-card {
            background: linear-gradient(135deg,#0E7C7B 0%, #0A5E5D 100%);
            color: #FFFFFF; border-radius: 16px; padding: 22px 26px;
            box-shadow: 0 4px 16px rgba(14,124,123,0.25);
        }
        .pred-card .big { font-size: 3rem; font-weight: 800; line-height: 1; }
        .pred-card .lbl { font-size: 0.95rem; opacity: .9; }

        /* Pestañas un poco más grandes */
        button[data-baseweb="tab"] { font-size: 0.98rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(titulo, descripcion=""):
    """Encabezado de sección consistente (IHC: jerarquía + reconocer)."""
    st.markdown(
        f"<div class='sec-head'><h3>{titulo}</h3>"
        f"<p>{descripcion}</p></div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# Carga de modelos y datos  (informe sección 2 — preparación)
# =============================================================================
@st.cache_resource(show_spinner="Cargando modelos entrenados...")
def load_models():
    """Carga el modelo, el scaler, el label encoder y el metadata (.pkl).

    Usa @st.cache_resource para no recargar los objetos en cada interacción
    (instrucción adicional 2). Reporta los archivos faltantes (instrucción 5).
    """
    archivos = {
        "model": "mejor_modelo_rf.pkl",
        "scaler": "scaler.pkl",
        "le": "label_encoder.pkl",
        "metadata": "metadata.pkl",
    }
    objetos, faltantes = {}, []
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
    """Carga el CSV y normaliza nombres y tipos de columna.

    - Limpia espacios sobrantes en los nombres ('Life expectancy ', ' BMI ').
    - Fuerza tipos numéricos y deja Status como texto plano para evitar
      problemas con los dtypes 'Arrow' de pandas reciente en Streamlit Cloud
      (que dejaban vacío el multiselect y descuadraban el slider).
    Usa @st.cache_data (instrucción adicional 2).
    """
    df = pd.read_csv(source)
    df.columns = df.columns.str.strip()
    if "Status" in df.columns:
        df["Status"] = df["Status"].astype(str).str.strip()
    for col in df.columns:
        if col not in ("Country", "Status"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_dataframe():
    """Carga del dataset con respaldos (informe sección 2).

    1) CSV local del repositorio  →  2) URL raw de GitHub  →  3) file_uploader.
    """
    if os.path.exists(DATA_PATH):
        try:
            return load_data_from_source(DATA_PATH)
        except Exception:
            pass
    try:
        return load_data_from_source(URL_DATASET)
    except Exception:
        st.warning(
            "No se pudo cargar el dataset automáticamente. "
            "Sube el CSV manualmente para continuar."
        )
    archivo = st.file_uploader("Sube life_expectancy.csv", type=["csv"])
    if archivo is not None:
        return load_data_from_source(archivo)
    return None


# -----------------------------------------------------------------------------
# Inicialización
# -----------------------------------------------------------------------------
inject_css()

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

    Usa el LabelEncoder entrenado; si falla, aplica el mapeo del plan
    (Developed=0, Developing=1).
    """
    try:
        return int(le.transform([valor_texto])[0])
    except Exception:
        return 0 if valor_texto == "Developed" else 1


# =============================================================================
# Encabezado general + navegación
# =============================================================================
st.title("🌍 Expectativa de Vida Global")
st.caption(
    "Análisis y Predicción · Dataset WHO · 193 países · 2000–2015 · "
    "Modelo: Random Forest Regressor"
)

st.sidebar.markdown("### 🧭 Navegación")
panel = st.sidebar.radio(
    "Selecciona un panel",
    ["📊 Panel A — Análisis de datos", "🔮 Panel B — Predicción"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")


# =============================================================================
# PANEL A — ANÁLISIS DE DATOS  (informe sección 3.1 · rúbrica 1.5 pts)
# =============================================================================
def render_panel_a():
    section_header(
        "📊 Panel A — Análisis exploratorio interactivo",
        "Explora la relación entre desarrollo, economía y longevidad. "
        "Usa los filtros de la izquierda para acotar la muestra.",
    )

    # ---- Filtros en el sidebar (informe sección 3.1) -----------------------
    st.sidebar.markdown("### 🔎 Filtros")

    anios = sorted(int(a) for a in df["Year"].dropna().unique())
    anio_sel = st.sidebar.selectbox("Año", ["Todos"] + anios, index=0)

    status_opts = sorted(df["Status"].dropna().astype(str).unique().tolist())
    status_sel = st.sidebar.multiselect(
        "Status", status_opts, default=status_opts
    )

    le_min = float(np.nanmin(df[TARGET]))
    le_max = float(np.nanmax(df[TARGET]))
    rango_le = st.sidebar.slider(
        "Rango de expectativa de vida (años)",
        min_value=float(round(le_min, 1)),
        max_value=float(round(le_max, 1)),
        value=(float(round(le_min, 1)), float(round(le_max, 1))),
    )

    # ---- Aplicar filtros ---------------------------------------------------
    dff = df.copy()
    if anio_sel != "Todos":
        dff = dff[dff["Year"] == anio_sel]
    if status_sel:
        dff = dff[dff["Status"].isin(status_sel)]
    dff = dff[(dff[TARGET] >= rango_le[0]) & (dff[TARGET] <= rango_le[1])]
    dff = dff.dropna(subset=[TARGET])

    # Visibilidad del estado del sistema: resumen de filtros activos
    resumen = (
        f"Año: **{anio_sel}** · "
        f"Status: **{', '.join(status_sel) if status_sel else 'ninguno'}** · "
        f"Expectativa: **{rango_le[0]:.0f}–{rango_le[1]:.0f} años**"
    )
    st.caption("🎛️ Filtros activos — " + resumen)

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
        delta=f"{pais_max['Country']} · {int(pais_max['Year'])}",
        delta_color="off",
    )
    k4.metric(
        "Menor expectativa",
        f"{pais_min[TARGET]:.1f} años",
        delta=f"{pais_min['Country']} · {int(pais_min['Year'])}",
        delta_color="off",
    )
    st.markdown("")

    # ---- Visualizaciones en pestañas (IHC: agrupación + menos scroll) ------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💰 GDP vs Vida", "🔗 Correlaciones", "📦 Por Status", "📈 Evolución"]
    )

    # Gráfico A1 — Scatter GDP vs Expectativa de vida
    with tab1:
        section_header(
            "A1 · GDP per cápita vs Expectativa de vida",
            "Cada punto es un país-año. Tamaño = años de escolaridad. "
            "Eje X en escala logarítmica.",
        )
        dff_scatter = dff.dropna(subset=["GDP", TARGET, "Schooling"])
        fig_a1 = px.scatter(
            dff_scatter,
            x="GDP",
            y=TARGET,
            color="Status",
            size="Schooling",
            log_x=True,
            color_discrete_map=COLOR_MAP,
            hover_data=["Country", "Year", TARGET, "GDP"],
            labels={"GDP": "GDP per cápita (log)", TARGET: "Expectativa (años)"},
            opacity=0.75,
        )
        fig_a1.update_layout(height=520, legend_title_text="Status",
                             margin=dict(t=20, r=10, b=10, l=10))
        st.plotly_chart(fig_a1, use_container_width=True)

    # Gráfico A2 — Mapa de calor de correlación
    with tab2:
        section_header(
            "A2 · Matriz de correlación de Pearson",
            "Verde = relación positiva, rojo = negativa. "
            "Útil para detectar las variables más ligadas a la longevidad.",
        )
        num_cols = dff.select_dtypes(include=[np.number]).drop(
            columns=["Year"], errors="ignore"
        )
        corr = num_cols.corr(numeric_only=True).round(2)
        fig_a2 = px.imshow(
            corr,
            color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1, text_auto=True, aspect="auto",
        )
        fig_a2.update_layout(height=650, margin=dict(t=20, r=10, b=10, l=10))
        st.plotly_chart(fig_a2, use_container_width=True)

    # Gráfico A3 — Boxplot por Status
    with tab3:
        section_header(
            "A3 · Distribución de la expectativa por Status",
            "Compara la mediana, el rango y los valores atípicos (outliers) "
            "entre países desarrollados y en desarrollo.",
        )
        fig_a3 = px.box(
            dff,
            x="Status", y=TARGET, color="Status",
            points="outliers", color_discrete_map=COLOR_MAP,
            labels={TARGET: "Expectativa de vida (años)"},
        )
        fig_a3.update_layout(height=520, showlegend=False,
                             margin=dict(t=20, r=10, b=10, l=10))
        st.plotly_chart(fig_a3, use_container_width=True)

    # Gráfico A4 — Serie temporal
    with tab4:
        section_header(
            "A4 · Evolución temporal por Status",
            "Promedio anual de expectativa de vida (2000–2015). "
            "Muestra la brecha entre ambos grupos a lo largo del tiempo.",
        )
        serie = (
            dff.groupby(["Year", "Status"])[TARGET].mean()
            .reset_index().sort_values("Year")
        )
        fig_a4 = px.line(
            serie,
            x="Year", y=TARGET, color="Status", markers=True,
            color_discrete_map=COLOR_MAP,
            labels={TARGET: "Expectativa promedio (años)", "Year": "Año"},
        )
        fig_a4.update_layout(height=520, margin=dict(t=20, r=10, b=10, l=10))
        st.plotly_chart(fig_a4, use_container_width=True)


# =============================================================================
# PANEL B — PREDICCIÓN  (informe sección 3.2 · rúbrica 1.5 pts)
# =============================================================================
def slider_feature(label, feature, fmt="%.2f"):
    """Slider para una feature usando min/median/max del metadata (instr. 3)."""
    vmin = float(FEATURE_MIN[feature])
    vmax = float(FEATURE_MAX[feature])
    vmed = min(max(float(FEATURE_MEDIAN[feature]), vmin), vmax)
    if vmin == vmax:
        vmax = vmin + 1.0
    return st.slider(label, min_value=vmin, max_value=vmax, value=vmed, format=fmt)


def gauge_resultado(pred, prom_global, le_min, le_max):
    """Medidor visual del resultado (IHC: feedback inmediato y comprensible)."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pred,
            number={"suffix": " años", "font": {"size": 40}},
            delta={"reference": prom_global, "suffix": " vs mundial"},
            gauge={
                "axis": {"range": [le_min, le_max]},
                "bar": {"color": COLOR_DEVELOPED},
                "steps": [
                    {"range": [le_min, 60], "color": "#F6D6C8"},
                    {"range": [60, 72], "color": "#FBEFD3"},
                    {"range": [72, le_max], "color": "#D6EFE2"},
                ],
                "threshold": {
                    "line": {"color": "#C0392B", "width": 3},
                    "thickness": 0.8, "value": prom_global,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=30, r=20, b=10, l=20))
    return fig


def render_panel_b():
    section_header(
        "🔮 Panel B — Predicción de expectativa de vida",
        "Define los indicadores de un país hipotético y el modelo "
        "Random Forest estimará su expectativa de vida.",
    )

    # Promedios de referencia para contextualizar el resultado
    prom_global = float(df[TARGET].mean())
    prom_status = df.groupby("Status")[TARGET].mean().to_dict()
    le_min = float(np.nanmin(df[TARGET]))
    le_max = float(np.nanmax(df[TARGET]))

    # ---- Entradas del usuario dentro de un formulario (IHC: feedback) ------
    # st.form agrupa todos los controles y ejecuta una sola vez al enviar.
    with st.form("form_prediccion"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 🩺 Mortalidad")
            adult_mortality = slider_feature("Adult Mortality (por 1000 adultos)", "Adult Mortality")
            infant_deaths = slider_feature("Infant deaths (por 1000 nacimientos)", "infant deaths")
            under_five = slider_feature("Under-five deaths (por 1000 nacimientos)", "under-five deaths")

        with col2:
            st.markdown("##### 💰 Salud y economía")
            gdp = slider_feature("GDP (USD per cápita)", "GDP")
            total_exp = slider_feature("Total expenditure (% gasto en salud)", "Total expenditure")
            hiv = slider_feature("HIV/AIDS (muertes por 1000 hab.)", "HIV/AIDS")

        with col3:
            st.markdown("##### 🎓 Desarrollo")
            schooling = slider_feature("Schooling (años de escolaridad)", "Schooling")
            icor = slider_feature("Income composition of resources (0-1)", "Income composition of resources")
            status_txt = st.selectbox("Status", ["Developing", "Developed"])

        st.markdown("")
        enviar = st.form_submit_button(
            "🔮 Predecir expectativa de vida", type="primary",
            use_container_width=True,
        )

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

    # ---- Flujo de predicción (informe sección 3.2) ------------------------
    if enviar:
        with st.spinner("Calculando predicción..."):
            # 1) Vector en el orden EXACTO de metadata['features'] (instr. 4).
            #    Las features no expuestas usan la mediana del dataset (instr. 3).
            fila, fuera_de_rango = [], []
            for feat in FEATURES:
                if feat in user_values:
                    val = float(user_values[feat])
                    if feat != "Status" and (
                        val < float(FEATURE_MIN[feat]) or val > float(FEATURE_MAX[feat])
                    ):
                        fuera_de_rango.append(feat)
                else:
                    val = float(FEATURE_MEDIAN[feat])
                fila.append(val)

            X = np.array(fila, dtype=float).reshape(1, -1)
            X_scaled = scaler.transform(X)          # 2) escalar
            pred = float(model.predict(X_scaled)[0])  # 3) predecir

        # ---- 4) Mostrar el resultado --------------------------------------
        st.markdown("")
        if fuera_de_rango:
            st.warning(
                "Algunos valores están fuera del rango observado en el dataset; "
                "la predicción puede ser menos confiable: "
                + ", ".join(f"`{f}`" for f in fuera_de_rango)
            )

        diff_global = pred - prom_global
        prom_grp = prom_status.get(status_txt, prom_global)
        diff_grp = pred - prom_grp

        col_res, col_gauge = st.columns([1, 1.1])

        with col_res:
            st.markdown(
                f"<div class='pred-card'>"
                f"<div class='lbl'>Expectativa de vida estimada</div>"
                f"<div class='big'>{pred:.1f} años</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")
            m1, m2 = st.columns(2)
            m1.metric(
                "vs promedio mundial", f"{prom_global:.1f}",
                delta=f"{diff_global:+.1f} años",
            )
            m2.metric(
                f"vs promedio {status_txt}", f"{prom_grp:.1f}",
                delta=f"{diff_grp:+.1f} años",
            )

        with col_gauge:
            st.plotly_chart(
                gauge_resultado(pred, prom_global, le_min, le_max),
                use_container_width=True,
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
    else:
        st.info(
            "Ajusta los indicadores y pulsa **Predecir** para obtener la "
            "estimación. Las variables no mostradas usan la mediana del dataset."
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
