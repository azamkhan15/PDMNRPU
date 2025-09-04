"""
myapp25.py

PdM Wind Turbine Dashboard — minimal targeted fix for top-table alignment issue.

Changes vs myapp24:
- Render top-left and top-right tables using Streamlit's st.columns() inside the same placeholder
  container instead of relying on a single flex HTML wrapper. This prevents the right table from
  dropping under the left table after Pause/Restart/resume.
- Slightly reduce table font-size and cap left pane width to reduce width pressure.
- Preserve all dynamic logic (EMA, demo/model RUL, graphs, beep, per-sensor blinking).
"""

from typing import Dict, Tuple, Any, List, Optional
import logging
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import platform
from pathlib import Path
from keras.models import load_model

# --- APPEND ONLY: suppress noisy TF/Keras warnings (keeps logs clean, no behavior change) ---
import warnings
warnings.filterwarnings("ignore", message="No training configuration found in the save file")
warnings.filterwarnings("ignore", message="Compiled the loaded model")

# Optional model import, protected by try/except at runtime
try:
    from tensorflow.keras.models import load_model  # type: ignore
except Exception:
    load_model = None  # type: ignore

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Myapp25")

# ---------------- Configuration ----------------
CSV_PATH = Path("simulated_wind_turbine_data_with_noise2.csv")
MODEL_PATH = "cnn_lstm_wind_turbine_rul.h5"

model = load_model(MODEL_PATH, compile=False)
EMA_ALPHA = 0.25
SEQUENCE_LENGTH = 10
GRAPH_UPDATE_STEP = 3

# Demo Controls
DEMO_TIME_SEC = 30
START_RUL = 6.25
WARNING_RUL = 6.0
BEEP_RUL = 5.75
RETURN_RUL = 5.5

SPEED_MAP = {"Fast": 0.015, "Medium": 0.1, "Slow": 0.25}

FEATURE_NAMES: List[str] = [
    "ambient_temperature_c", "wind_speed_mps", "humidity_percent",
    "rpm", "output_power_w", "generator_voltage_v",
    "generator_current_a", "noise_db", "vibration_mmps"
]

DEMO_STEP_DELAY = SPEED_MAP["Fast"]
DEMO_LENGTH = max(10, int(DEMO_TIME_SEC / DEMO_STEP_DELAY))

NORMAL_RANGES: Dict[str, Tuple[float, float]] = {
    "ambient_temperature_c": (5.0, 40.0),
    "wind_speed_mps": (0.0, 15.0),
    "humidity_percent": (20.0, 80.0),
    "rpm": (100.0, 400.0),
    "output_power_w": (0.0, 800.0),
    "generator_voltage_v": (20.0, 30.0),
    "generator_current_a": (0.0, 50.0),
    "noise_db": (20.0, 80.0),
    "vibration_mmps": (0.0, 3.0),
}
NORMAL_MARGIN_FRAC = 0.10

# ---------------- Helpers ----------------
def safe_float(value: Any, default: float = float("nan")) -> float:
    """Safely convert a value to float, return default on failure.

    Args:
        value: Any input value.
        default: Default float to return when conversion fails.

    Returns:
        Converted float or default.
    """
    try:
        return float(value)
    except Exception:
        return default


def ema_update(prev: Optional[float], new: float, alpha: float = EMA_ALPHA) -> float:
    """Exponential moving average update.

    Args:
        prev: Previous EMA value or None.
        new: New raw sample value.
        alpha: Smoothing factor.

    Returns:
        Updated EMA as float.
    """
    if prev is None or (isinstance(prev, float) and np.isnan(prev)):
        return float(new)
    return float(alpha * new + (1 - alpha) * prev)


def generate_demo_data(length: int = 600) -> pd.DataFrame:
    """Generate an oscillatory synthetic dataset (same as previous versions).

    Args:
        length: Number of rows.

    Returns:
        DataFrame with the defined FEATURES.
    """
    t = np.linspace(0, 10 * np.pi, length)
    return pd.DataFrame({
        "ambient_temperature_c": 20 + 10 * np.sin(t),
        "wind_speed_mps": 6 + 2 * np.sin(t * 0.8),
        "humidity_percent": 40 + 15 * np.sin(t * 1.2),
        "rpm": 200 + 80 * np.sin(t * 0.5),
        "output_power_w": 600 + 100 * np.sin(t * 0.3),
        "generator_voltage_v": 24 + 4 * np.sin(t * 0.7),
        "generator_current_a": 25 + 8 * np.sin(t * 0.9),
        "noise_db": 30 + 25 * np.sin(t * 1.5),
        "vibration_mmps": 0.3 + 1.0 * np.sin(t * 2)
    })


def generate_demo_rul(idx: int,
                      length: int = DEMO_LENGTH,
                      start_val: float = START_RUL,
                      min_val: float = RETURN_RUL,
                      up_val: float = START_RUL) -> float:
    """Generate demo RUL with fast-down / slow-up cycle."""
    down_len = max(1, int(length * 0.35))
    up_len = max(1, length - down_len)
    cycle_pos = idx % length
    if cycle_pos < down_len:
        frac = cycle_pos / down_len
        rul = start_val - (start_val - min_val) * frac
    else:
        frac = (cycle_pos - down_len) / up_len
        rul = min_val + (up_val - min_val) * frac
    return float(rul)


def get_health_status(rul: float) -> str:
    """Return health label based on RUL thresholds."""
    if rul > WARNING_RUL:
        return "✅ Good"
    elif rul >= BEEP_RUL:
        return "⚠️ Warning"
    else:
        return "❌ Critical"


def sensor_health_label_and_level(sensor: str, value: float) -> Tuple[str, str]:
    """Return sensor health label and CSS level ('good'|'warning'|'critical')."""
    if sensor == "rul":
        if value > WARNING_RUL:
            return "✅ Good", "good"
        elif value >= BEEP_RUL:
            return "⚠️ Warning", "warning"
        else:
            return "❌ Critical", "critical"

    rng = NORMAL_RANGES.get(sensor)
    if rng is None:
        return "✅ Good", "good"
    low, high = rng
    span = high - low if high - low != 0 else 1.0
    margin = span * NORMAL_MARGIN_FRAC
    if low <= value <= high:
        return "✅ Good", "good"
    elif (low - margin) <= value < low or high < value <= (high + margin):
        return "⚠️ Warning", "warning"
    else:
        return "❌ Critical", "critical"


def system_beep() -> None:
    """Optional OS beep (commented out by default)."""
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 500)
        elif platform.system() == "Linux":
            import os
            os.system('play -nq -t alsa synth 0.1 sine 1000')
    except Exception:
        # Non-critical; swallow errors for portability
        logger.debug("system_beep failed", exc_info=True)


# ---------------- App layout ----------------
st.set_page_config(page_title="PdM Wind Turbine Dashboard", layout="wide")
st.markdown(
    "<h3 style='text-align:center;color:#fff;background:#1f2937;padding:10px;border-radius:8px'>"
    "Predictive Maintenance – Small Urban Wind Turbine (600 W)  Version 1.0 </h3>",
    #"<h5 style='text-align:center;color:#fff;background:#1f2937;padding:10px;border-radius:8px'>"
    #" Note: Demo Version: START_RUL = 6.25 WARNING_RUL = 6.0 BEEP_RUL = 5.75 RETURN_RUL = 5.5</h5>"
    unsafe_allow_html=True
)


# ---------------- Load model safely ----------------
model = None
if load_model is not None:
    try:
        model = load_model(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    except Exception as exc:
        model = None
        logger.exception("Failed to load model: %s", exc)
        st.error(f"Could not load model at {MODEL_PATH}. Model-based predictions will be disabled.")
else:
    logger.info("Keras load_model not available in this environment; model-based predictions disabled.")

# ---------------- Session State ----------------
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "running" not in st.session_state:
    st.session_state.running = False
if "ema" not in st.session_state:
    st.session_state.ema = {f: None for f in FEATURE_NAMES + ["rul"]}
if "hist" not in st.session_state:
    st.session_state.hist = {f: [] for f in FEATURE_NAMES + ["rul"]}
if "seq_buffer" not in st.session_state:
    st.session_state.seq_buffer = []
if "delay" not in st.session_state:
    st.session_state.delay = SPEED_MAP["Fast"]
if "buzzer_on" not in st.session_state:
    st.session_state.buzzer_on = False
if "mode" not in st.session_state:
    st.session_state.mode = "Demo Simulation"

# ---------------- Controls ----------------
col_mode, col_speed = st.columns([2, 1])
with col_mode:
    st.session_state.mode = st.radio("Select Mode:", ["Real Simulation", "Demo Simulation"], index=1)
with col_speed:
    speed = st.selectbox("Speed", list(SPEED_MAP.keys()), index=0)
    st.session_state.delay = SPEED_MAP[speed]

c1, c2, c3 = st.columns(3)
if c1.button("Start"):
    st.session_state.running = True
if c2.button("Pause"):
    st.session_state.running = False
if c3.button("Reset"):
    st.session_state.running = False
    st.session_state.idx = 0
    st.session_state.hist = {f: [] for f in FEATURE_NAMES + ["rul"]}
    st.session_state.ema = {f: None for f in FEATURE_NAMES + ["rul"]}
    st.session_state.seq_buffer = []
    st.session_state.buzzer_on = False
    try:
        top_placeholder.empty()
    except Exception:
        pass

# ---------------- Sidebar Graph Selection ----------------
st.sidebar.markdown("### 📊 Graph Visibility")
show_rul_graph = st.sidebar.checkbox("Show RUL over Time", value=True)
show_feature_graphs = st.sidebar.multiselect("Select Features to Plot", FEATURE_NAMES, default=FEATURE_NAMES)

# ---------------- Placeholders ----------------
top_placeholder = st.empty()
placeholder_metrics = st.empty()
placeholder_graphs = st.empty()
placeholder_rul = st.empty()
placeholder_beep = st.empty()
placeholder_timer = st.empty()

# ---------------- CSS (injected once) ----------------
TOP_TABLE_CSS = """
<style>
/* Keep badges & blinking as before */
.badge-good { color:#065f46; background:#ecfdf5; padding:4px 6px; border-radius:6px; display:inline-block; font-size:12px; }
.badge-warning { color:#92400e; background:#fffbeb; padding:4px 6px; border-radius:6px; display:inline-block; font-size:12px; }
.badge-critical { color:#7f1d1d; background:#fff1f2; padding:4px 6px; border-radius:6px; display:inline-block; font-size:12px; }
@keyframes blink {0% {background-color:#fff1f2;}50%{background-color:#ffb4b4;}100%{background-color:#fff1f2;}}
.blink { animation: blink 1s linear infinite; }

/* Table styling: slightly smaller font to reduce width pressure */
.sensor-table { border-collapse: collapse; width: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; font-size:13px; box-sizing: border-box; }
.sensor-table th { background: linear-gradient(90deg, #1f2937, #374151); color: #fff; padding: 6px 8px; text-align: left; font-size:12px; }
.sensor-table td { padding: 6px 8px; background: #ffffff; font-size:12px; color: #0f172a; border-bottom: 1px solid #e6e9ee; word-break: break-word; }

/* Cap the left pane width to avoid it stealing space on reflows */
.top-left { max-width: 420px; box-sizing: border-box; min-width: 0; }
.top-right { min-width: 0; box-sizing: border-box; }
</style>
"""
if not st.session_state.get("_top_css_injected", False):
    st.markdown(TOP_TABLE_CSS, unsafe_allow_html=True)
    st.session_state["_top_css_injected"] = True

# ---------------- Data ----------------
if st.session_state.mode == "Demo Simulation":
    df = generate_demo_data(DEMO_LENGTH)
else:
    @st.cache_data
    def load_csv(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except Exception as exc:
            logger.exception("Failed to load CSV %s: %s", path, exc)
            raise
    try:
        df = load_csv(CSV_PATH)
    except Exception:
        df = pd.DataFrame(columns=FEATURE_NAMES)
        st.error(f"Could not load CSV at {CSV_PATH}, using empty DataFrame.")

# ---------------- Top-table HTML builders (return left and right HTML fragments) ----------------
def build_left_table_html(ema: Dict[str, Optional[float]]) -> str:
    """Return HTML for the left 'Sensor Data' table."""
    current_vals = {f: (ema.get(f) if ema.get(f) is not None else float("nan")) for f in FEATURE_NAMES}
    current_vals["rul"] = ema.get("rul") if ema.get("rul") is not None else START_RUL
    html = "<div class='top-left'><table class='sensor-table'><thead><tr><th colspan='2'>Sensor Data</th></tr></thead><tbody>"
    display_order = [
        ("Wind Speed", "wind_speed_mps"),
        ("Temperature", "ambient_temperature_c"),
        ("Humidity", "humidity_percent"),
        ("RPM", "rpm"),
        ("Output Power (W)", "output_power_w"),
        ("Generator Voltage (V)", "generator_voltage_v"),
        ("Generator Current (A)", "generator_current_a"),
        ("Noise (dB)", "noise_db"),
        ("Vibration (mm/s)", "vibration_mmps"),
        ("RUL (days)", "rul"),
    ]
    for label, key in display_order:
        val = current_vals.get(key, float("nan"))
        display_val = f"{val:.2f}" if not (isinstance(val, float) and np.isnan(val)) else "—"
        html += f"<tr><td><strong>{label}</strong></td><td>{display_val}</td></tr>"
    html += "</tbody></table></div>"
    return html


def build_right_table_html(ema: Dict[str, Optional[float]]) -> str:
    """Return HTML for the right 'Sensor Health' table."""
    current_vals = {f: (ema.get(f) if ema.get(f) is not None else float("nan")) for f in FEATURE_NAMES}
    current_vals["rul"] = ema.get("rul") if ema.get("rul") is not None else START_RUL
    html = "<div class='top-right'><table class='sensor-table'><thead><tr><th>Sensor</th><th>Normal Range</th><th>Current Value</th><th>Health</th></tr></thead><tbody>"
    friendly = {
        "ambient_temperature_c": "Temperature (°C)",
        "wind_speed_mps": "Wind Speed (m/s)",
        "humidity_percent": "Humidity (%)",
        "rpm": "RPM",
        "output_power_w": "Power (W)",
        "generator_voltage_v": "Generator Voltage (V)",
        "generator_current_a": "Generator Current (A)",
        "noise_db": "Noise (dB)",
        "vibration_mmps": "Vibration (mm/s)",
        "rul": "RUL (days)"
    }
    for sensor in FEATURE_NAMES + ["rul"]:
        val = current_vals.get(sensor, float("nan"))
        display_val = f"{val:.2f}" if not (isinstance(val, float) and np.isnan(val)) else "—"
        if sensor == "rul":
            rng_text = f"> {WARNING_RUL}"
        else:
            rng = NORMAL_RANGES.get(sensor)
            rng_text = f"{rng[0]:.2f}–{rng[1]:.2f}" if rng else "—"
        label, level = sensor_health_label_and_level(sensor, val if not (isinstance(val, float) and np.isnan(val)) else float("nan"))
        badge_class = "badge-good" if level == "good" else ("badge-warning" if level == "warning" else "badge-critical")
        blink_class = "blink" if level == "critical" else ""
        health_cell = f"<div class='{badge_class} {blink_class}'>{label}</div>"
        value_td_class = "class='" + blink_class + "'" if blink_class else ""
        html += f"<tr><td><strong>{friendly.get(sensor, sensor)}</strong></td><td>{rng_text}</td><td {value_td_class}>{display_val}</td><td>{health_cell}</td></tr>"
    html += "</tbody></table></div>"
    return html

# ---------------- Initial render of top tables (so user sees tables even when not running) ----------------
with top_placeholder.container():
    left_html = build_left_table_html(st.session_state.ema)
    right_html = build_right_table_html(st.session_state.ema)
    col_left, col_right = st.columns([0.45, 0.55])
    col_left.markdown(left_html, unsafe_allow_html=True)
    col_right.markdown(right_html, unsafe_allow_html=True)

# ---------------- Main Loop ----------------
while st.session_state.running and st.session_state.idx < len(df):
    row = df.iloc[st.session_state.idx]
    features = [safe_float(row.get(f, 0)) for f in FEATURE_NAMES]

    st.session_state.seq_buffer.append(features)
    if len(st.session_state.seq_buffer) > SEQUENCE_LENGTH:
        st.session_state.seq_buffer.pop(0)

    # Predict / Demo RUL
    if st.session_state.mode == "Demo Simulation":
        rul_value = generate_demo_rul(st.session_state.idx, DEMO_LENGTH)
    else:
        rul_value = None
        if model is not None and len(st.session_state.seq_buffer) == SEQUENCE_LENGTH:
            try:
                X_input = np.array(st.session_state.seq_buffer).reshape(1, SEQUENCE_LENGTH, len(FEATURE_NAMES))
                rul_value = float(model.predict(X_input, verbose=0))
            except Exception as exc:
                logger.exception("Model prediction failed: %s", exc)
                rul_value = None

    # EMA updates
    for i, f in enumerate(FEATURE_NAMES):
        st.session_state.ema[f] = ema_update(st.session_state.ema[f], safe_float(row.get(f, 0)))
        st.session_state.hist[f].append(st.session_state.ema[f])
    rul_input = rul_value if rul_value is not None else st.session_state.ema["rul"] or START_RUL
    st.session_state.ema["rul"] = ema_update(st.session_state.ema["rul"], rul_input)
    st.session_state.hist["rul"].append(st.session_state.ema["rul"])

    # ---------------- Top tables: use columns inside the same container to avoid wrapping ----------------
    with top_placeholder.container():
        left_html = build_left_table_html(st.session_state.ema)
        right_html = build_right_table_html(st.session_state.ema)
        col_left, col_right = st.columns([0.45, 0.55])
        col_left.markdown(left_html, unsafe_allow_html=True)
        col_right.markdown(right_html, unsafe_allow_html=True)

    # ---------------- Metrics (only RUL + health) ----------------
    with placeholder_metrics.container():
        health_status = get_health_status(st.session_state.ema["rul"])
        st.markdown(f"<h4 style='text-align:center;'>RUL (days): {st.session_state.ema['rul']:.2f} | {health_status}</h4>",
                    unsafe_allow_html=True)

    # ---------------- Timer (demo) ----------------
    if st.session_state.mode == "Demo Simulation":
        elapsed = st.session_state.idx * st.session_state.delay
        remaining = max(0, DEMO_TIME_SEC - elapsed)
        placeholder_timer.markdown(
            f"<div style='text-align:center;font-size:14px;color:#1f2937'>⏳ Demo ends in {remaining:.1f}s</div>",
            unsafe_allow_html=True
        )

    # ---------------- Graphs ----------------
    if st.session_state.idx % GRAPH_UPDATE_STEP == 0:
        with placeholder_graphs.container():
            if show_feature_graphs:
                rows = int(np.ceil(len(show_feature_graphs) / 3))
                fig, axs = plt.subplots(rows, 3, figsize=(9, 3 * rows))
                axs = axs.flatten()
                for i, f in enumerate(show_feature_graphs):
                    ax = axs[i]
                    ax.plot(st.session_state.hist[f], color='tab:blue')
                    ax.set_title(f, fontsize=10)
                    ax.grid(True, linestyle="--", alpha=0.3)
                for j in range(i + 1, len(axs)):
                    axs[j].axis("off")
                plt.tight_layout()
                st.pyplot(fig)
                # --- APPEND ONLY: close figure after rendering to avoid "too many open figures" ---
                plt.close(fig)
        with placeholder_rul.container():
            if show_rul_graph:
                st.markdown("<h4>Predicted RUL over Time</h4>", unsafe_allow_html=True)
                fig_rul, ax_rul = plt.subplots(figsize=(12, 4))
                ax_rul.plot(st.session_state.hist["rul"], color='red', linewidth=2)
                ax_rul.set_ylabel("RUL (days)")
                ax_rul.set_xlabel("Time Step")
                ax_rul.grid(True, linestyle="--", alpha=0.3)
                st.pyplot(fig_rul)
                # --- APPEND ONLY: close figure after rendering ---
                plt.close(fig_rul)

    # ---------------- Beep (unchanged behavior) ----------------
    if st.session_state.ema["rul"] <= BEEP_RUL:
        if not st.session_state.buzzer_on:
            st.session_state.buzzer_on = True
            # system_beep()  # uncomment for local OS beep
        placeholder_beep.markdown(
            """<audio autoplay loop>
               <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg">
               </audio>""",
            unsafe_allow_html=True
        )
    else:
        st.session_state.buzzer_on = False
        placeholder_beep.empty()

    st.session_state.idx += 1
    time.sleep(st.session_state.delay)
