"""
TradeMatch AI — Streamlit Frontend Entry Point
"""

import streamlit as st

st.set_page_config(
    page_title="TradeMatch AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌐 TradeMatch AI")
st.subheader("Swipe-to-Export Intelligent Matchmaking")

st.markdown(
    """
    Welcome to **TradeMatch AI** — an intelligent matchmaking platform
    that connects exporters with the most compatible importers using
    live trade signals, global news sentiment, and ML-powered scoring.

    👈 Use the sidebar to navigate between pages.
    """
)
