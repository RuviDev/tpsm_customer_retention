import base64
import json
import re
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from utils.ui_helpers import page_header, section_title, info_card


ROOT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT_DIR / "notebooks"
DATA_DIR = ROOT_DIR / "data"

DIMENSION_CONFIG = {
    "Dimension 1": {
        "notebook": NOTEBOOK_DIR / "Dim1-Des.Stat.ipynb",
        "dataset": DATA_DIR / "dimension_01_dataset.csv",
        "description": "Responsiveness & Support Speed descriptive analysis.",
    },
    "Dimension 2": {
        "notebook": NOTEBOOK_DIR / "Dim2-Des.Stat.ipynb",
        "dataset": DATA_DIR / "dimension_02_dataset.csv",
        "description": "Interaction Experience descriptive analysis.",
    },
}


def get_theme_mode() -> str:
    return st.session_state.get("theme_mode", "Dark")


def get_chart_colors():
    theme = get_theme_mode()

    if theme == "Dark":
        return {
            "bg": "#0b1120",
            "panel": "#111827",
            "text": "#f8fafc",
            "muted": "#cbd5e1",
            "grid": "#334155",
        }

    return {
        "bg": "#f8fafc",
        "panel": "#ffffff",
        "text": "#0f172a",
        "muted": "#334155",
        "grid": "#cbd5e1",
    }


def style_plot(fig, ax):
    colors = get_chart_colors()

    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["panel"])

    ax.title.set_color(colors["text"])
    ax.xaxis.label.set_color(colors["text"])
    ax.yaxis.label.set_color(colors["text"])

    ax.tick_params(axis="x", colors=colors["text"], labelcolor=colors["text"])
    ax.tick_params(axis="y", colors=colors["text"], labelcolor=colors["text"])

    for label in ax.get_xticklabels():
        label.set_color(colors["text"])

    for label in ax.get_yticklabels():
        label.set_color(colors["text"])

    for spine in ax.spines.values():
        spine.set_color(colors["grid"])

    ax.grid(axis="y", linestyle="--", alpha=0.35, color=colors["grid"])

    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor(colors["panel"])
        legend.get_frame().set_edgecolor(colors["grid"])

        for text in legend.get_texts():
            text.set_color(colors["text"])


def load_notebook(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_code_cell(cells: list, marker: str) -> dict | None:
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))

        if marker in source:
            return cell

    return None


def parse_text_table(text: str | None) -> pd.DataFrame | None:
    if not text:
        return None

    raw = text.strip("\n")

    if not raw:
        return None

    lines = [line.rstrip() for line in raw.splitlines() if line.strip()]

    if not lines:
        return None

    if any(line.startswith("dtype:") for line in lines):
        rows = []

        for line in lines:
            if line.startswith("dtype:"):
                continue

            parts = line.rsplit(maxsplit=1)

            if len(parts) != 2:
                continue

            key, value = parts

            try:
                parsed = float(value)
                if parsed.is_integer():
                    parsed = int(parsed)
            except ValueError:
                parsed = value

            rows.append((key, parsed))

        if rows:
            return pd.DataFrame(rows, columns=["Column", "Value"]).set_index("Column")

    try:
        df = pd.read_fwf(StringIO(raw))
    except Exception:
        return None

    if df.empty:
        return None

    return df


def get_first_html_table(cell: dict, table_index: int = 0) -> pd.DataFrame | None:
    tables = []

    for output in cell.get("outputs", []):
        if output.get("output_type") not in {"display_data", "execute_result"}:
            continue

        data = output.get("data", {})
        html = data.get("text/html")

        if html is None:
            text = data.get("text/plain")

            if isinstance(text, list):
                text = "".join(text)

            parsed_text = parse_text_table(text)

            if parsed_text is not None:
                tables.append(parsed_text)

            continue

        if isinstance(html, list):
            html = "".join(html)

        try:
            parsed = pd.read_html(StringIO(html))
            tables.extend(parsed)
        except Exception:
            text = data.get("text/plain")

            if isinstance(text, list):
                text = "".join(text)

            parsed_text = parse_text_table(text)

            if parsed_text is not None:
                tables.append(parsed_text)

    if not tables or table_index >= len(tables):
        return None

    df = tables[table_index]

    if len(df.columns) > 0 and str(df.columns[0]).startswith("Unnamed"):
        df = df.set_index(df.columns[0])
        df.index.name = None

    return df


def extract_image_b64(cell: dict) -> list[str]:
    images = []

    for output in cell.get("outputs", []):
        if output.get("output_type") not in {"display_data", "execute_result"}:
            continue

        data = output.get("data", {})
        image_data = data.get("image/png")

        if image_data is None:
            continue

        if isinstance(image_data, list):
            image_data = "".join(image_data)

        images.append(image_data)

    return images


def extract_dataset_path(cells: list) -> str | None:
    pattern = re.compile(r"file_path\s*=\s*['\"]([^'\"]+\.csv)['\"]")

    for cell in cells:
        source = "".join(cell.get("source", []))
        match = pattern.search(source)

        if match:
            return match.group(1)

    return None


def resolve_local_dataset(config_dataset_path: Path, notebook_dataset_path: str | None) -> Path | None:
    if config_dataset_path.exists():
        return config_dataset_path

    if notebook_dataset_path:
        raw = Path(notebook_dataset_path)

        if raw.exists():
            return raw

        matches = sorted(DATA_DIR.rglob(raw.name))

        if matches:
            return matches[0]

        matches = sorted(ROOT_DIR.rglob(raw.name))

        if matches:
            return matches[0]

    return None


def central_tendency(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    mean_values = df[numeric_cols].mean(numeric_only=True)
    median_values = df[numeric_cols].median(numeric_only=True)
    mode_values = df[numeric_cols].mode().transpose()
    mode_values.columns = [f"Mode_{i + 1}" for i in range(mode_values.shape[1])]

    summary = pd.DataFrame(
        {
            "Mean": mean_values,
            "Median": median_values,
        }
    )

    summary = summary.merge(mode_values, left_index=True, right_index=True, how="left")

    return summary


def variability(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    q1 = df[numeric_cols].quantile(0.25)
    q3 = df[numeric_cols].quantile(0.75)

    return pd.DataFrame(
        {
            "Range": df[numeric_cols].max() - df[numeric_cols].min(),
            "Variance": df[numeric_cols].var(),
            "Standard Deviation": df[numeric_cols].std(),
            "IQR": q3 - q1,
        }
    )


def frequency_table(series: pd.Series) -> pd.DataFrame:
    counts = series.value_counts(dropna=False)
    freq_df = counts.rename("Count").to_frame()
    freq_df["Percentage"] = (freq_df["Count"] / len(series) * 100).round(2)
    freq_df.index = freq_df.index.astype("object").where(freq_df.index.notna(), "Missing")
    freq_df.index.name = series.name

    return freq_df


def render_histograms(df: pd.DataFrame, numeric_cols: list[str]):
    st.subheader("Histograms")

    cols = st.columns(2, gap="large")

    for idx, col in enumerate(numeric_cols):
        with cols[idx % 2]:
            try:
                fig, ax = plt.subplots(figsize=(5.2, 3.4))
                sns.histplot(df[col].dropna(), kde=True, bins=24, ax=ax, color="#38bdf8")

                ax.set_title(col.replace("_", " ").title(), fontsize=12)
                ax.set_xlabel(col.replace("_", " ").title(), fontsize=10)
                ax.set_ylabel("Frequency", fontsize=10)

                style_plot(fig, ax)
                fig.tight_layout()

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            except Exception as exc:
                st.warning(f"Could not create histogram for '{col}': {exc}")


def render_boxplots(df: pd.DataFrame, numeric_cols: list[str]):
    st.subheader("Boxplots")

    cols = st.columns(2, gap="large")

    for idx, col in enumerate(numeric_cols):
        with cols[idx % 2]:
            try:
                fig, ax = plt.subplots(figsize=(5.0, 3.4))
                sns.boxplot(y=df[col].dropna(), ax=ax, color="#67a8ff", width=0.35)

                ax.set_title(col.replace("_", " ").title(), fontsize=12)
                ax.set_ylabel(col.replace("_", " ").title(), fontsize=10)

                style_plot(fig, ax)
                fig.tight_layout()

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            except Exception as exc:
                st.warning(f"Could not create boxplot for '{col}': {exc}")


def render_pie_charts(df: pd.DataFrame, categorical_cols: list[str]):
    st.subheader("Pie Charts")

    colors = get_chart_colors()
    cols = st.columns(2, gap="large")

    for idx, col in enumerate(categorical_cols):
        with cols[idx % 2]:
            try:
                value_counts = df[col].value_counts(dropna=False)

                if len(value_counts) > 9:
                    top = value_counts.head(9)
                    other_count = value_counts.iloc[9:].sum()
                    final_counts = pd.concat([top, pd.Series({"Other": other_count})]) if other_count > 0 else top
                else:
                    final_counts = value_counts

                final_counts.index = final_counts.index.astype("object").where(
                    final_counts.index.notna(),
                    "Missing",
                )

                fig, ax = plt.subplots(figsize=(5.0, 4.2))

                wedges, texts, autotexts = ax.pie(
                    final_counts,
                    labels=final_counts.index,
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={
                        "fontsize": 8,
                        "color": colors["text"],
                    },
                )

                for text in texts:
                    text.set_color(colors["text"])

                for autotext in autotexts:
                    autotext.set_color(colors["text"])
                    autotext.set_fontweight("bold")

                ax.set_title(
                    col.replace("_", " ").title(),
                    fontsize=12,
                    color=colors["text"],
                )

                ax.axis("equal")
                fig.patch.set_facecolor(colors["bg"])
                ax.set_facecolor(colors["panel"])

                fig.tight_layout()

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            except Exception as exc:
                st.warning(f"Could not create pie chart for '{col}': {exc}")

def render_notebook_images(images: list[str], title: str):
    st.subheader(title)

    if not images:
        st.info(f"No notebook images found for {title.lower()}.")
        return

    cols = st.columns(2, gap="large")

    for idx, img_b64 in enumerate(images):
        with cols[idx % 2]:
            st.image(base64.b64decode(img_b64), use_container_width=True)


def render_dimension_dashboard(config: dict, dimension_label: str):
    notebook_path = config["notebook"]
    configured_dataset_path = config["dataset"]

    nb = load_notebook(notebook_path)
    cells = nb.get("cells", [])

    notebook_dataset_path = extract_dataset_path(cells) if cells else None
    local_dataset_path = resolve_local_dataset(configured_dataset_path, notebook_dataset_path)

    df = None

    if local_dataset_path is not None:
        try:
            df = pd.read_csv(local_dataset_path)
        except Exception as exc:
            st.warning(f"Dataset found but could not be loaded: {exc}")
            df = None

    st.markdown(f"### {dimension_label}")
    st.caption(config["description"])

    if notebook_path.exists():
        st.caption(f"Notebook Source: `{notebook_path.name}`")
    else:
        st.caption("Notebook Source: Not found. Showing dataset-based calculations only if dataset exists.")

    if local_dataset_path is not None:
        st.caption(f"Dataset Source: `{local_dataset_path.name}`")
    else:
        st.caption("Dataset Source: Not found. Trying to use notebook outputs if available.")

    st.divider()

    if df is not None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    else:
        numeric_cols = []
        categorical_cols = []

    cell_central = find_code_cell(cells, "Measures of Central Tendency") or {}
    cell_variability = find_code_cell(cells, "Measures of Variability") or {}
    cell_hist = find_code_cell(cells, "histplot") or {}
    cell_box = find_code_cell(cells, "boxplot") or {}
    cell_pie = find_code_cell(cells, "plt.pie") or {}

    section_title("Section 1: Measures of Central Tendency")

    if df is not None:
        if numeric_cols:
            ct_df = central_tendency(df, numeric_cols).round(6)
            st.dataframe(ct_df, use_container_width=True)
        else:
            st.info("No numeric columns available.")
    else:
        ct_df = get_first_html_table(cell_central, table_index=0)

        if ct_df is not None:
            st.dataframe(ct_df, use_container_width=True)
        else:
            st.info("Central tendency table not available.")

    st.divider()

    section_title("Section 2: Measures of Variability")

    if df is not None:
        if numeric_cols:
            var_df = variability(df, numeric_cols).round(6)
            st.dataframe(var_df, use_container_width=True)
        else:
            st.info("No numeric columns available.")
    else:
        var_df = get_first_html_table(cell_variability, table_index=0)

        if var_df is not None:
            st.dataframe(var_df, use_container_width=True)
        else:
            st.info("Variability table not available.")

    st.divider()

    section_title("Section 3: Frequency Distribution and Visualizations")

    if df is not None:
        if categorical_cols:
            st.subheader("Categorical Frequency Tables")

            for col in categorical_cols:
                with st.expander(col.replace("_", " ").title(), expanded=False):
                    st.dataframe(frequency_table(df[col]), use_container_width=True)
        else:
            st.info("No categorical columns available for frequency tables.")

        if numeric_cols:
            render_histograms(df, numeric_cols)
            render_boxplots(df, numeric_cols)

        if categorical_cols:
            render_pie_charts(df, categorical_cols)

    else:
        st.info("Showing chart outputs captured from the notebook execution.")
        render_notebook_images(extract_image_b64(cell_hist), "Histograms")
        render_notebook_images(extract_image_b64(cell_box), "Boxplots")
        render_notebook_images(extract_image_b64(cell_pie), "Pie Charts")


page_header(
    "Descriptive Statistics Dashboard",
    "Interactive descriptive analytics dashboard for Dimension 1 and Dimension 2."
)

st.markdown(
    """
    This page presents the descriptive analytics part of the assignment.  
    It summarizes central tendency, variability, frequency distributions, and visualizations for both selected dimensions.
    """
)

col1, col2 = st.columns(2)

with col1:
    info_card(
        "Dimension 1",
        "Responsiveness & Support Speed. This dimension focuses on response time, CSAT score, support channel, issue category, and related service speed variables."
    )

with col2:
    info_card(
        "Dimension 2",
        "Interaction Experience. This dimension focuses on customer sentiment, politeness, empathy, helpfulness, tone, and outcome quality."
    )

tab_dim1, tab_dim2 = st.tabs(["Dimension 1", "Dimension 2"])

with tab_dim1:
    render_dimension_dashboard(DIMENSION_CONFIG["Dimension 1"], "Dimension 1")

with tab_dim2:
    render_dimension_dashboard(DIMENSION_CONFIG["Dimension 2"], "Dimension 2")

st.divider()
st.caption("TPSM Project - Descriptive Statistics Dashboard")