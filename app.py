import os, requests, streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("🩺 Clinical Text Extractor")
st.caption("Structured oncology variables from clinical notes — traced in Langfuse.")

text = st.text_area("Paste clinical text:", height=160)

if st.button("Extract") and text.strip():
    with st.spinner("Extracting..."):
        resp = requests.post(f"{API_URL}/extract", json={"text": text}, timeout=60)
        resp.raise_for_status()
    st.json(resp.json())