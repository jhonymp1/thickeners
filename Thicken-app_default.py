
# Thicken-app_default.py — Pearson-only correlations, bilingual ES/EN UI, loads default Excel if available
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import re
import io
from pathlib import Path

# === Utilidad simple para mostrar ES/EN ===
def t(es: str, en: str) -> str:
    return f"{es} / {en}"

# ===============================
# Config de la app
# ===============================
st.set_page_config(page_title="Thicken-app", layout="wide")
st.title("Thicken-app")
st.caption(t(
    "Trabaja con los nombres originales del Excel; solo cambian si los renombras en la UI. Remuestreo toma el primer valor del intervalo.",
    "Works with original Excel column names; they only change if you rename them in the UI. Resampling takes the first value of each interval."
))

# ===============================
# Sidebar: Parámetros
# ===============================
with st.sidebar:
    st.header(t("⚙️ Parámetros", "⚙️ Settings"))
    freq = st.text_input(
        t("Frecuencia de remuestreo (p.ej., 10T, 5T, 30T, 1H)", "Resampling frequency (e.g., 10T, 5T, 30T, 1H)"),
        value="10T"
    )
    use_resampled_for_corr = st.checkbox(t("Usar datos remuestreados para correlación", "Use resampled data for correlation"), value=True)
    min_frac = st.slider(t("Mínimo de datos válidos por columna (fracción)", "Minimum valid data per column (fraction)"), 0.05, 0.8, 0.20, 0.05)
    show_diagnostics = st.checkbox(t("Mostrar diagnósticos", "Show diagnostics"), value=False)

    st.markdown("---")
    st.subheader(t("Correlación móvil (opcional)", "Rolling correlation (optional)"))
    enable_rolling = st.checkbox(t("Activar correlación móvil", "Enable rolling correlation"), value=False)
    rolling_minutes = st.number_input(t("Ventana (minutos)", "Window (minutes)"), min_value=5, max_value=240, value=60, step=5)

    st.markdown("---")
    uploaded = st.file_uploader("📤 " + t("Cargar Excel", "Upload Excel"), type=["xlsx", "xls"])

# ===============================
# Utilidades
# ===============================

def is_unnamed(col) -> bool:
    c = str(col).strip().lower()
    return (c == "") or c.startswith("unnamed") or (c in {"nan", "none"})

def detect_header_row(raw: pd.DataFrame, candidatos: list[str], max_rows: int = 40) -> int | None:
    for r in range(min(max_rows, len(raw))):
        fila = raw.iloc[r].astype(str).str.strip().str.lower()
        hits = sum(any(cand.lower() in v for v in fila) for cand in candidatos)
        if hits >= 2:
            return r
    return None

def build_columns_from_row(raw: pd.DataFrame, header_row: int) -> pd.Index:
    return (
        raw.iloc[header_row]
        .ffill(axis=0)
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

def normalize_column_names(cols: pd.Index) -> pd.Index:
    return cols.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

def drop_ghost_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    first_data_row = df.iloc[0]
    first_row_empty_mask = first_data_row.replace(r"^\s*$", np.nan, regex=True).isna()
    tmp_all = df.replace(r"^\s*$", np.nan, regex=True)
    all_empty_mask = tmp_all.isna().all(axis=0)
    drop_mask = first_row_empty_mask & all_empty_mask
    return df.loc[:, ~drop_mask]

def to_numeric_locale_aware(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if not out[col].dtype == "O":
            continue
        ser = out[col].astype(str)
        ser = ser.str.replace("\u00A0", "", regex=False)
        ser = ser.str.replace("%", "", regex=False).str.strip()
        sample = ser.dropna().head(200)
        cnt_comma = sample.str.contains(",", regex=False).sum()
        cnt_dot = sample.str.contains(r"\.", regex=True).sum()
        cnt_both = sample.str.contains(r".*[,].*[.].*|.*[.].*[,].*", regex=True).sum()
        if cnt_both > 0 or (cnt_comma > cnt_dot and cnt_comma > 0):
            ser = ser.str.replace(r"\.", "", regex=True)
            ser = ser.str.replace(",", ".", regex=False)
        else:
            ser = ser.str.replace(r"(?<=\d)[\s']", "", regex=True)
            ser = ser.str.replace(r"[^0-9\.\-+eE]", "", regex=True)
        out[col] = pd.to_numeric(ser, errors="coerce")
    return out

def diagnose_numeric(df: pd.DataFrame, label: str):
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        st.warning(t("[DIAG] No hay columnas numéricas.", "[DIAG] No numeric columns."))
        return num
    stats = pd.DataFrame({
        "non_null": num.notna().sum(),
        "nulls": num.isna().sum(),
        "std": num.std(ddof=0),
        "min": num.min(),
        "max": num.max(),
    }).sort_values("non_null", ascending=False)
    with st.expander(t("🔍 Diagnóstico numérico", "🔍 Numeric diagnostics") + f": {label}", expanded=False):
        st.write(t(f"Columnas numéricas ({num.shape[1]}): {list(num.columns)}", f"Numeric columns ({num.shape[1]}): {list(num.columns)}"))
        st.dataframe(stats)
    return num

# === Pearson solamente ===

def compute_correlations(num_df: pd.DataFrame, min_frac: float = 0.2):
    if num_df.empty:
        return None
    min_valid = max(3, int(min_frac * len(num_df)))
    valid_cols = []
    for c in num_df.columns:
        ser = num_df[c].dropna()
        if ser.size >= min_valid and ser.std(ddof=0) > 1e-12:
            valid_cols.append(c)
    num_df = num_df[valid_cols]
    if len(valid_cols) < 2:
        return None
    pear = num_df.corr(method="pearson", min_periods=min_valid)
    if pear is not None:
        pear.index.name = None
        pear.columns.name = None
    return pear

# ==== utilidades para descarga de figuras ====

def _sanitize_filename(text: str) -> str:
    import re as _re
    ttxt = _re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text or "fig"))
    return ttxt[:120]

def fig_to_png_bytes(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    return buf

# Heatmap único (Pearson)

def plot_one_heatmap(mat: pd.DataFrame, title_suffix=""):
    fig, ax = plt.subplots(figsize=(12, 9))
    if mat is None or mat.isna().all().all():
        ax.set_title(t("Pearson: sin datos suficientes", "Pearson: not enough data"), fontsize=14)
        ax.axis("off")
        st.pyplot(fig)
        return fig
    mask = np.triu(np.ones_like(mat, dtype=bool))
    with sns.axes_style("white"):
        sns.heatmap(
            mat, mask=mask, annot=True, fmt=".2f",
            cmap="magma", linewidths=0.5, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax, annot_kws={"size": 11, "color": "black"}
        )
    ax.set_title(t("Pearson", "Pearson") + f" {title_suffix}", fontsize=16, pad=10)
    plt.tight_layout()
    st.pyplot(fig)
    return fig


def top_abs_corr_dataframe(mat: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    if mat is None or getattr(mat, "empty", True):
        return pd.DataFrame(columns=["Var 1", "Var 2", "corr"])
    m = mat.copy().astype(float)
    mask = np.triu(np.ones(m.shape, dtype=bool), k=1)
    m = m.abs().where(mask)
    s = m.stack().dropna()
    s.index = s.index.set_names(["Var 1", "Var 2"])
    s.name = "corr"
    df_top = (
        s.reset_index()
        .sort_values("corr", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )
    return df_top


def _make_unique(names: list[str]) -> list[str]:
    seen = {}
    out = []
    for n in names:
        base = (n or "").strip()
        if base == "":
            base = "col"
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


def find_time_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "Date","date","Fecha","fecha","Date Time","date time",
        "Timestamp","timestamp","Time","time","Datetime","datetime"
    ]
    for c in df.columns:
        if str(c) in candidates:
            ts = pd.to_datetime(df[c], errors="coerce")
            if ts.notna().sum() >= 3:
                return c
    for c in df.columns:
        ts = pd.to_datetime(df[c], errors="coerce")
        if ts.notna().sum() >= max(3, int(0.2*len(df))):
            return c
    return None

# ===============================
# Lógica principal
# ===============================
# Si no hay archivo subido, intentamos cargar el Excel por defecto
default_path = Path("Datos practicando.xlsx")
if uploaded is None and default_path.exists():
    st.info(t("Se cargó el archivo por defecto: Datos practicando.xlsx", "Loaded default file: Datos practicando.xlsx"))
    uploaded = str(default_path)

if uploaded is None:
    st.info(t("Carga un archivo Excel para comenzar.", "Upload an Excel file to start."))
    st.stop()

# 1) Leer crudo
try:
    src = uploaded  # puede ser st.UploadedFile o ruta str
    raw = pd.read_excel(src, header=None, engine="openpyxl")
except Exception as e:
    st.error(t(f"No se pudo leer el Excel: {e}", f"Could not read Excel: {e}"))
    st.stop()

# 2) Detectar encabezado
candidatos = ["Date Time","Torque","Sólidos","Flujo","Bed","Claridad","Dosis","Fecha"]
header_row = detect_header_row(raw, candidatos=candidatos, max_rows=40)
if header_row is None:
    st.error(t("No se pudo detectar la fila de encabezado en las primeras 40 filas. Ajusta el archivo o amplía la búsqueda.",
               "Header row not found within the first 40 rows. Adjust the file or widen the search."))
    st.stop()

st.success(t(f"Fila de encabezado detectada (índice pandas): {header_row}", f"Header row detected (pandas index): {header_row}"))

# 3) Columnas y df base (NOMBRES ORIGINALES LIMPIOS)
cols = build_columns_from_row(raw, header_row=header_row)
df = (raw.iloc[header_row + 1:].copy().pipe(lambda d: d.set_axis(cols, axis=1)))

# 4) Limpieza de nombres y columnas no válidas (SIN traducir ni mapear)
df = (df.loc[:, ~df.columns.map(is_unnamed)]
      .pipe(lambda d: d.set_axis(normalize_column_names(d.columns), axis=1))
      .pipe(drop_ghost_columns))

# Estado de sesión
original_cols_clean = list(df.columns)
signature = tuple(original_cols_clean)
if ("df_signature" not in st.session_state) or (st.session_state.df_signature != signature):
    st.session_state.df_signature = signature
    st.session_state.col_names_current = original_cols_clean.copy()
    st.session_state.time_col = None
    st.session_state.vars_plot = []
    st.session_state.var_sec = "(ninguna)"

df = df.set_axis(st.session_state.col_names_current, axis=1)

# 5) Detectar columna de tiempo
if st.session_state.time_col is None or st.session_state.time_col not in df.columns:
    time_col = find_time_column(df)
    if time_col is None:
        st.error(t("No se encontró una columna de tiempo reconocible (p. ej., 'Date', 'Date Time', 'Fecha').",
                   "No recognizable time column found (e.g., 'Date', 'Date Time', 'Fecha')."))
        st.stop()
    st.session_state.time_col = time_col

time_col = st.session_state.time_col

# Convertir a datetime y ordenar
df = df.copy()
df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.dropna(subset=[time_col]).sort_values(time_col)

# ===============================
# Sidebar: Renombrar columnas
# ===============================
with st.sidebar:
    st.subheader(t("✏️ Renombrar columnas", "✏️ Rename columns"))
    st.caption(t("Los nombres visibles son los **originales/actuales**. Solo cambian al pulsar **Aplicar**.",
                 "Visible names are the **original/current** ones. They only change when you press **Apply**."))

    allow_rename_time = st.checkbox(
        t(f"Permitir renombrar la columna de tiempo ('{time_col}')", f"Allow renaming the time column ('{time_col}')"), value=False,
        help=t("No recomendado; podría afectar el remuestreo y los gráficos.", "Not recommended; it may affect resampling and charts.")
    )

    current_cols_now = list(df.columns)
    col1, col2 = st.columns(2)
    with col1:
        reset_names = st.button("↩️ " + t("Reset", "Reset"), use_container_width=True)
    with col2:
        apply_renames = st.button("✅ " + t("Aplicar", "Apply"), type="primary", use_container_width=True)

    default_names = st.session_state.col_names_current.copy()
    if reset_names:
        default_names = original_cols_clean.copy()

    if not allow_rename_time:
        idx_time = current_cols_now.index(time_col)
        default_names[idx_time] = st.session_state.col_names_current[idx_time]

    typed_names = default_names.copy()
    for i, col in enumerate(current_cols_now):
        if (not allow_rename_time) and (col == time_col):
            st.text_input(f"{col} " + t("(bloqueada)", "(locked)"), value=default_names[i], disabled=True, key=f"rename_{i}")
            typed_names[i] = default_names[i]
            continue
        typed_names[i] = st.text_input(f"{col}", value=default_names[i], key=f"rename_{i}")

if apply_renames:
    new_names = _make_unique([x.strip() for x in typed_names])
    if not allow_rename_time:
        idx_time = list(df.columns).index(time_col)
        new_names[idx_time] = df.columns[idx_time]
    mapping = {old: new for old, new in zip(df.columns, new_names) if old != new}
    if mapping:
        df = df.set_axis(new_names, axis=1)
        st.session_state.col_names_current = new_names
        if allow_rename_time and time_col in mapping:
            st.session_state.time_col = mapping[time_col]
        time_col = st.session_state.time_col
        st.success(t(f"Columnas renombradas: {mapping}", f"Renamed columns: {mapping}"))
    else:
        st.info(t("No hubo cambios en los nombres de columnas.", "No column name changes."))

# 6) Conversión numérica
df_num_crudo = to_numeric_locale_aware(df)

# 7) Remuestreo (primer valor del intervalo)
df_resampled = (
    df_num_crudo
    .set_index(time_col)
    .resample(freq)
    .first()
    .reset_index()
)

# ===============================
# Serie de tiempo
# ===============================
st.subheader(t("📊 Serie de tiempo", "📊 Time series"))
plot_df = df_resampled if time_col in df_resampled.columns else df_num_crudo
num_cols_plot = [c for c in plot_df.columns if c != time_col and pd.api.types.is_numeric_dtype(plot_df[c])]

options_key = ("plot_options_key", tuple(num_cols_plot), time_col)
if "plot_options_key" not in st.session_state or st.session_state.plot_options_key != options_key:
    st.session_state.plot_options_key = options_key
    prev_vars = st.session_state.get("vars_plot", [])
    st.session_state.vars_plot = [c for c in prev_vars if c in num_cols_plot] or (num_cols_plot[:2] if len(num_cols_plot) >= 2 else num_cols_plot[:1])
    prev_sec = st.session_state.get("var_sec", "(ninguna)")
    st.session_state.var_sec = prev_sec if prev_sec in num_cols_plot else "(ninguna)"

vars_plot = st.multiselect(
    t("Variables a graficar (eje primario)", "Variables to plot (primary axis)"),
    options=num_cols_plot,
    default=st.session_state.vars_plot,
    key="vars_plot"
)

var_sec = st.selectbox(
    t("Variable en eje secundario (opcional)", "Secondary-axis variable (optional)"),
    options=["(ninguna)"] + num_cols_plot,
    index=(["(ninguna)"] + num_cols_plot).index(st.session_state.var_sec) if st.session_state.var_sec in (["(ninguna)"] + num_cols_plot) else 0,
    key="var_sec"
)

if vars_plot or (var_sec != "(ninguna)"):
    y_cols = vars_plot.copy()
    if var_sec != "(ninguna)" and var_sec not in y_cols:
        y_cols.append(var_sec)
    if y_cols:
        fig_ts = px.line(
            plot_df, x=time_col, y=y_cols,
            title=t("Serie de tiempo (remuestreado)", "Time series (resampled)") if plot_df is df_resampled else t("Serie de tiempo (crudo)", "Time series (raw)")
        )
        if var_sec != "(ninguna)":
            fig_ts.update_traces(yaxis="y2", selector=dict(name=var_sec))
            fig_ts.update_layout(
                yaxis=dict(title=t("Eje primario", "Primary axis")),
                yaxis2=dict(title=var_sec, overlaying="y", side="right")
            )
        st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info(t("Selecciona al menos una variable para graficar.", "Select at least one variable to plot."))

# ===============================
# Histogramas
# ===============================
st.subheader(t("Histogramas (exploración de distribución)", "Histograms (distribution exploration)"))
st.caption(t(
    "Selecciona columnas numéricas. El origen crudo/remuestreado sigue el switch de la barra lateral (\"Usar datos remuestreados para correlación\").",
    "Select numeric columns. Source (raw/resampled) follows the sidebar switch (\"Use resampled data for correlation\")."
))

try:
    _df_num = df_num_crudo.copy()
except NameError:
    _df_num = df.copy()
try:
    _df_res = df_resampled.copy()
except NameError:
    _df_res = None

try:
    use_resampled_hist = bool(use_resampled_for_corr)
except NameError:
    use_resampled_hist = True

if use_resampled_hist and _df_res is not None and st.session_state.get('time_col', None) in _df_res.columns:
    df_base_hist = _df_res
    st.caption(t("Base usada: remuestreado", "Base used: resampled"))
else:
    df_base_hist = _df_num
    st.caption(t("Base usada: crudo", "Base used: raw"))

num_candidates = [c for c in df_base_hist.columns if c != st.session_state.get('time_col', None) and pd.api.types.is_numeric_dtype(df_base_hist[c])]

with st.expander(t("Configuración de gráfico", "Chart settings"), expanded=False):
    cols_sel = st.multiselect(t("Columnas numéricas", "Numeric columns"), options=num_candidates, default=num_candidates[:1])
    bins_method = st.selectbox(t("Método de bins", "Bin method"), options=["auto","fd","sturges","scott","sqrt","rice"], index=0)
    show_kde = st.checkbox(t("Mostrar KDE (curva de densidad)", "Show KDE (density curve)"), value=True)
    show_vlines = st.checkbox(t("Mostrar media / mediana / moda", "Show mean / median / mode"), value=True)
    use_log = st.checkbox(t("Escala logarítmica (x)", "Log scale (x)"), value=False)

from numpy import histogram_bin_edges as _bin_edges

def _hist_series(series, title=None, idx: int = 0):
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        st.info(t(f"Sin datos válidos para {getattr(series, 'name', 'serie')}", f"No valid data for {getattr(series, 'name', 'series')}")); return
    try:
        edges = _bin_edges(s.values, bins=bins_method)
    except Exception:
        edges = _bin_edges(s.values, bins='auto')

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.histplot(s, bins=edges, kde=show_kde, ax=ax)

    if show_vlines:
        mu=s.mean(); med=s.median()
        try:
            mode_val=s.mode().iloc[0]
        except Exception:
            mode_val=None
        ax.axvline(mu, ls='--', c='k', label=t("media=","mean=")+f"{mu:.4g}")
        ax.axvline(med, ls='--', c='b', label=t("mediana=","median=")+f"{med:.4g}")
        if mode_val is not None:
            ax.axvline(mode_val, ls='--', c='g', label=t("moda=","mode=")+f"{mode_val:.4g}")

    if use_log:
        try: ax.set_xscale('log')
        except Exception: pass

    ax.set_title(title or f"Distribución / Distribution – {getattr(series, 'name', '')}"); ax.grid(True, linestyle=':', alpha=0.5)
    if show_vlines: ax.legend()
    st.pyplot(fig)

    fname = f"hist_{_sanitize_filename(getattr(series, 'name', 'serie'))}.png"
    buf = fig_to_png_bytes(fig)
    st.download_button(t("⬇️ Descargar histograma (PNG)", "⬇️ Download histogram (PNG)"), data=buf, file_name=fname, mime="image/png", key=f"dl_hist_{idx}")

    q = s.quantile([0.05,0.25,0.5,0.75,0.95])
    stats_df = pd.DataFrame({'métrica':['count','missing','mean','std','min','p5','p25','median','p75','p95','max','skew','kurtosis'],
                             'valor':[int(s.count()), int(series.size - s.count()), s.mean(), s.std(ddof=0), s.min(), q.get(0.05), q.get(0.25), q.get(0.5), q.get(0.75), q.get(0.95), s.max(), s.skew(), s.kurt()]})
    st.dataframe(stats_df, use_container_width=True)

if cols_sel:
    containers = st.columns(2) if len(cols_sel) > 1 else [st.container()]
    for i,c in enumerate(cols_sel):
        holder = containers[i % len(containers)]
        with holder:
            _hist_series(df_base_hist[c], title=f"{c}", idx=i)
else:
    st.info(t("Selecciona al menos una columna numérica en las opciones de histograma.", "Select at least one numeric column in histogram options."))

# ===============================
# 🔥 Mapa de calor de correlaciones (Pearson)
# ===============================
st.subheader(t("🔥 Mapa de calor de correlaciones (Pearson)", "🔥 Correlation heatmap (Pearson)"))
base_for_corr = df_resampled if use_resampled_for_corr else df_num_crudo
label = t("remuestreado", "resampled") if use_resampled_for_corr else t("crudo", "raw")

if show_diagnostics:
    _ = diagnose_numeric(base_for_corr, label)

num_df = base_for_corr.select_dtypes(include=[np.number])
pear = compute_correlations(num_df, min_frac=min_frac)

if pear is None:
    st.warning(t("No hay suficientes columnas numéricas con datos válidos para calcular correlaciones.",
                 "Not enough numeric columns with valid data to compute correlations."))
else:
    fig_heat = plot_one_heatmap(pear, title_suffix=f"({label})")
    fname_hm = f"heatmap_pearson_{_sanitize_filename(label)}.png"
    st.download_button(t("⬇️ Descargar mapa de calor (PNG)", "⬇️ Download heatmap (PNG)"), data=fig_to_png_bytes(fig_heat), file_name=fname_hm, mime="image/png", key="dl_heatmap")

    st.markdown("**" + t("Top correlaciones – Pearson", "Top correlations – Pearson") + "**")
    st.dataframe(top_abs_corr_dataframe(pear), use_container_width=True)

# ===============================
# Correlación móvil (opcional)
# ===============================
if enable_rolling:
    st.subheader(t("⏱️ Correlación móvil (ventana por tiempo)", "⏱️ Rolling correlation (time window)"))
    base = base_for_corr.copy()
    if time_col in base.columns:
        base = base.set_index(time_col)
    cand = [c for c in base.columns if pd.api.types.is_numeric_dtype(base[c])]
    if len(cand) < 2:
        st.warning(t("Se requieren al menos 2 columnas numéricas para correlación móvil.", "At least 2 numeric columns are required for rolling correlation."))
    else:
        c1 = st.selectbox(t("Variable 1", "Variable 1"), options=cand, index=0)
        c2 = st.selectbox(t("Variable 2", "Variable 2"), options=cand, index=1 if len(cand) > 1 else 0)
        if c1 and c2 and c1 != c2:
            sel = base[[c1, c2]].dropna()
            try:
                rolling_corr = sel[c1].rolling(f"{int(rolling_minutes)}min").corr(sel[c2])
                fig2, ax2 = plt.subplots(figsize=(12, 4))
                ax2.plot(rolling_corr.index, rolling_corr.values, color="tab:blue")
                ax2.axhline(0, color='k', lw=0.8)
                ax2.set_title(t(f"Correlación móvil {c1} vs {c2} (ventana={rolling_minutes} min) – {label}",
                                 f"Rolling correlation {c1} vs {c2} (window={rolling_minutes} min) – {label}"))
                ax2.set_ylim(-1, 1)
                ax2.set_ylabel("Corr")
                st.pyplot(fig2)
            except Exception as e:
                st.error(t(f"No se pudo calcular correlación móvil: {e}", f"Could not compute rolling correlation: {e}"))
        else:
            st.info(t("Selecciona dos variables distintas.", "Select two different variables."))

# ===============================
# Notas
# ===============================
with st.expander(t("ℹ️ Notas", "ℹ️ Notes")):
    st.markdown(
        f"""
- {t('Los nombres originales del Excel se preservan y **solo cambian** si pulsas **Aplicar** en *Renombrar columnas*.',
       'Original Excel column names are preserved and **only change** after you press **Apply** in *Rename columns*.')}
- {t(f'El nombre vigente de tiempo es: **`{time_col}`** (puedes habilitar su renombre si lo necesitas).',
       f'Current time column is: **`{time_col}`** (you can enable renaming if needed).')}
- {t('El remuestreo toma el **primer valor** del intervalo (`.first()`).', 'Resampling takes the **first value** of the interval (`.first()`).')}
- {t('Las selecciones del gráfico permanecen aunque renombres (se guardan en sesión).', 'Chart selections persist across renames (stored in session).')}
- {t('Encontrarás botones para **descargar** los histogramas, el **mapa de calor (Pearson)** y la **gráfica de valores medidos vs. predichos** en PNG.',
       'You will find buttons to **download** histograms, the **heatmap (Pearson)**, and the **measured vs. predicted** chart as PNG.')}
        """
    )

# ===============================
# Modelado y exploración (dinámico)
# ===============================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

st.subheader(t("Modelado y exploración (dinámico)", "Modeling & exploration (dynamic)"))
st.caption(t("Elige objetivo y predictores desde el Excel; prueba modelo lineal, Random Forest o polinómico.",
            "Pick target and predictors from Excel; try linear, Random Forest, or polynomial models."))

try:
    df_base_model = base_for_corr.copy()
except NameError:
    df_base_model = df_num_crudo.copy() if 'df_num_crudo' in globals() else df.copy()

_suggestions = ['Torque','Bed Mass','Rake Lift','Flocculant','UF Flow','Tonnage','UF Density','Torque (%)','Masa de cama','Densidad UF','Caudal UF','Tonelaje','Rake Lift (%)']
cols_all = list(df_base_model.columns)
num_cols = [c for c in cols_all if pd.api.types.is_numeric_dtype(df_base_model[c])]

def _default_target():
    patt = re.compile(r"uf\s*dens|densidad\s*uf|solid\s*in\s*discharge|solidos.*descarga", re.I)
    for c in cols_all:
        if patt.search(str(c)):
            return c
    return num_cols[0] if num_cols else (cols_all[0] if cols_all else None)

if 'ml_target' not in st.session_state:
    st.session_state.ml_target = _default_target()
if 'ml_features' not in st.session_state:
    defaults = [c for c in cols_all if c != st.session_state.ml_target]
    order = sorted(defaults, key=lambda c: (0 if any(s.lower() in str(c).lower() for s in _suggestions) else 1, c))
    st.session_state.ml_features = order[:5]

left, right = st.columns(2)
with left:
    target_col = st.selectbox(t("Variable objetivo (y)", "Target variable (y)"), options=cols_all,
                              index=(cols_all.index(st.session_state.ml_target) if st.session_state.ml_target in cols_all else 0), key='ml_target')
with right:
    feature_cols = st.multiselect(t("Variables explicativas (X)", "Predictor variables (X)"), options=[c for c in cols_all if c != target_col],
                                  default=[c for c in st.session_state.ml_features if c in cols_all and c != target_col], key='ml_features')

with st.expander(t("Opciones de preprocesamiento", "Preprocessing options")):
    drop_na = st.checkbox(t("Eliminar filas con NA en columnas seleccionadas", "Drop rows with NA in selected columns"), value=True)
    standardize = st.checkbox(t("Estandarizar X (media 0, var 1) — solo OLS/polinómico", "Standardize X (mean 0, var 1) — OLS/polynomial only"), value=False)

with st.expander(t("Modelo", "Model")):
    model_name = st.selectbox(t("Tipo de modelo", "Model type"), options=["Lineal (OLS)", "Random Forest", "Polinómico (grado k)"], index=1)
    if model_name == "Random Forest":
        n_estimators = st.slider("n_estimators", 50, 600, 200, 25)
        max_depth = st.slider(t("max_depth (None = sin límite)", "max_depth (None = unlimited)"), 0, 40, 0, 1)
        max_depth = None if max_depth == 0 else max_depth
        rf_bootstrap = st.checkbox("bootstrap", True)
        rf_random_state = st.number_input("random_state", min_value=0, value=42, step=1)
    elif model_name == "Polinómico (grado k)":
        poly_degree = st.slider(t("Grado del polinomio", "Polynomial degree"), 2, 5, 2)
        include_bias = st.checkbox(t("Incluir bias en PolynomialFeatures", "Include bias in PolynomialFeatures"), False)

if target_col is None or not feature_cols:
    st.info(t("Selecciona una variable objetivo y al menos una variable explicativa.", "Select a target and at least one predictor."))
else:
    work = df_base_model[[target_col] + feature_cols].copy()

    if drop_na:
        work = work.dropna(subset=[target_col] + feature_cols)

    for c in [target_col] + feature_cols:
        if not pd.api.types.is_numeric_dtype(work[c]):
            work[c] = pd.to_numeric(work[c], errors='coerce')
    work = work.dropna(subset=[target_col] + feature_cols)

    if len(work) < 10:
        st.warning(t("No hay suficientes filas válidas después del preprocesamiento (se requieren ≥ 10).",
                     "Not enough valid rows after preprocessing (need ≥ 10)."))
    else:
        X = work[feature_cols]
        y = work[target_col]

        if standardize and model_name in {"Lineal (OLS)", "Polinómico (grado k)"}:
            mu = X.mean(axis=0); sigma = X.std(axis=0).replace(0, 1); Xs = (X - mu) / sigma
        else:
            Xs = X

        if model_name == "Lineal (OLS)":
            reg = LinearRegression(); reg.fit(Xs, y); pred_fn = (lambda Z: reg.predict((Z - mu) / sigma) if standardize else reg.predict(Z))
        elif model_name == "Random Forest":
            reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, bootstrap=rf_bootstrap, random_state=rf_random_state, n_jobs=-1)
            reg.fit(X, y); pred_fn = reg.predict
        else:
            poly = PolynomialFeatures(degree=poly_degree, include_bias=include_bias)
            from sklearn.pipeline import Pipeline
            if standardize:
                reg = Pipeline([("poly", poly), ("lin", LinearRegression())]); reg.fit(Xs, y); pred_fn = (lambda Z: reg.predict(((Z - mu) / sigma)))
            else:
                reg = Pipeline([("poly", poly), ("lin", LinearRegression())]); reg.fit(X, y); pred_fn = reg.predict

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
        if model_name == "Lineal (OLS)" and standardize:
            mu_tr = train_X.mean(axis=0); sigma_tr = train_X.std(axis=0).replace(0, 1)
            reg_e = LinearRegression().fit((train_X - mu_tr) / sigma_tr, train_y)
            preds = reg_e.predict((test_X - mu_tr) / sigma_tr)
        elif model_name == "Polinómico (grado k)":
            from sklearn.pipeline import Pipeline
            if standardize:
                mu_tr = train_X.mean(axis=0); sigma_tr = train_X.std(axis=0).replace(0, 1)
                pipe_e = Pipeline([("poly", PolynomialFeatures(degree=poly_degree, include_bias=include_bias)), ("lin", LinearRegression())]).fit((train_X - mu_tr) / sigma_tr, train_y)
                preds = pipe_e.predict((test_X - mu_tr) / sigma_tr)
            else:
                pipe_e = Pipeline([("poly", PolynomialFeatures(degree=poly_degree, include_bias=include_bias)), ("lin", LinearRegression())]).fit(train_X, train_y)
                preds = pipe_e.predict(test_X)
        else:
            preds = reg.predict(test_X)

        mse = mean_squared_error(test_y, preds); rmse = float(np.sqrt(mse)); r2 = r2_score(test_y, preds)
        st.markdown("**" + t("Resumen del modelo (hold-out 80/20):", "Model summary (hold-out 80/20):") + "**")
        st.write({t("Observaciones", "Observations"): len(X), t("Features", "Features"): len(feature_cols), t("Modelo", "Model"): model_name, "MSE": round(mse,3), "RMSE": round(rmse,3), "R2": round(r2,3)})

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(test_y, preds, alpha=0.7)
        min_v = min(test_y.min(), preds.min()); max_v = max(test_y.max(), preds.max())
        ax1.plot([min_v, max_v], [min_v, max_v], 'r--')
        ax1.set_xlabel(t(f"Valor real de {target_col}", f"Actual value of {target_col}")); ax1.set_ylabel(t(f"Valor predicho de {target_col}", f"Predicted value of {target_col}")); ax1.set_title(t("Comparación entre valores reales y predichos", "Actual vs predicted")); ax1.grid(True)
        st.pyplot(fig1)

        fname_sc = f"medidos_vs_predichos_{_sanitize_filename(target_col)}.png"
        st.download_button(t("⬇️ Descargar imagen (PNG)", "⬇️ Download image (PNG)"), data=fig_to_png_bytes(fig1), file_name=fname_sc, mime="image/png", key="dl_scatter")

        with st.expander(t("Predicción manual", "Manual prediction")):
            st.caption(t("Ingresa valores para las variables X. Los no especificados usarán la mediana del dataset.", "Enter values for X; unspecified use dataset median."))
            med = X.median(); p5 = X.quantile(0.05); p95 = X.quantile(0.95)
            user_vals = {}; cols_ui = st.columns(2) if len(feature_cols)>1 else [st.container()]
            for i, feat in enumerate(feature_cols):
                col = cols_ui[i % len(cols_ui)]
                with col:
                    vmin=float(p5.get(feat, X[feat].min())); vmax=float(p95.get(feat, X[feat].max())); vdef=float(med.get(feat, X[feat].median()))
                    step=(vmax-vmin)/100 if vmax>vmin else 0.01
                    user_vals[feat]=st.number_input(f"{feat}", value=round(vdef,4), min_value=float(min(vmin,vdef,vmax)), max_value=float(max(vmin,vdef,vmax)), step=step)
            if st.button(t("Calcular predicción", "Compute prediction"), type="primary"):
                row = pd.DataFrame([user_vals]); yhat = float(pred_fn(row))
                st.success(t(f"Predicción estimada para {target_col}: {yhat:.4f}", f"Estimated prediction for {target_col}: {yhat:.4f}"))
