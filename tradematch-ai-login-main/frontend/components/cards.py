"""
Reusable Streamlit card components for displaying exporter / importer profiles.
"""

import streamlit as st


def exporter_card(exporter: dict) -> None:
    """Render an exporter profile card."""
    with st.container():
        st.markdown(f"### 🏭 {exporter.get('name', 'Unknown Exporter')}")
        st.caption(f"Country: {exporter.get('country', 'N/A')}")
        st.write(exporter.get('description', ''))


def importer_card(importer: dict) -> None:
    """Render an importer profile card."""
    with st.container():
        st.markdown(f"### 🏪 {importer.get('name', 'Unknown Importer')}")
        st.caption(f"Country: {importer.get('country', 'N/A')}")
        st.write(importer.get('description', ''))
