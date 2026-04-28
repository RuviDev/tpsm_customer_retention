import streamlit as st
from pathlib import Path


def load_css():
    css_path = Path(__file__).resolve().parents[1] / "assets" / "style.css"

    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as file:
            st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(f'<div class="app-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def info_card(title, body):
    st.markdown(
        f"""
        <div class="custom-card">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_card(title, prediction_text, probability_text, risk_level):
    badge_class = "success-badge"

    if "Medium" in risk_level:
        badge_class = "warning-badge"

    if "High" in risk_level:
        badge_class = "danger-badge"

    st.markdown(
        f"""
        <div class="result-card">
            <h3>{title}</h3>
            <p><b>Prediction:</b> {prediction_text}</p>
            <p><b>Retained Probability:</b> {probability_text}</p>
            <span class="{badge_class}">{risk_level}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def locked_card(message):
    st.markdown(
        f"""
        <div class="locked-card">
            🔒 {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def method_box(text):
    clean_text = text.strip()

    st.html(
        f"""
        <div class="method-box">
            {clean_text}
        </div>
        """
    )