"""
Filter and search widgets for the Streamlit frontend.
"""

import streamlit as st


def country_filter(countries: list[str]) -> str | None:
    """Render a country filter dropdown. Returns the selected country or None."""
    return st.selectbox("Filter by Country", options=["All"] + sorted(countries))


def product_search() -> str:
    """Render a product keyword search box."""
    return st.text_input("Search by Product / HS Code")
