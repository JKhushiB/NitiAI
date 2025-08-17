import streamlit as st
print("API Key from secrets:", st.secrets.get("GROQ_API_KEY", "NOT FOUND"))