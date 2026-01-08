import streamlit as st
import spacy
from spacy import displacy
from spacy.cli import download

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="NER App",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Load spaCy Model
# -----------------------------
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# -----------------------------
# App Header
# -----------------------------
st.title("Named Entity Recognition (NER)")
st.write(
    "This application uses **spaCy** to identify entities such as "
    "**PERSON, ORG, GPE, DATE, LOC**, and more."
)

# -----------------------------
# Text Input
# -----------------------------
default_text = "Virat Kohli was born in Delhi and plays cricket for India."
text = st.text_area(
    "Enter text for entity recognition:",
    value=default_text,
    height=150
)

# -----------------------------
# Process Text
# -----------------------------
if st.button("Analyze Text"):
    if text.strip():
        doc = nlp(text)

        st.subheader("Detected Entities")

        if doc.ents:
            for ent in doc.ents:
                st.write(f"**{ent.text}** → `{ent.label_}`")
        else:
            st.warning("No entities found.")

        # -----------------------------
        # Visualization
        # -----------------------------
        st.subheader("Entity Visualization")
        html = displacy.render(doc, style="ent", jupyter=False)
        st.components.v1.html(html, height=300, scrolling=True)
    else:
        st.error("Please enter some text.")

# -----------------------------
# Sidebar Info
# -----------------------------
st.sidebar.header("About")
st.sidebar.info(
    """
    **NER App**
    
    - Built with spaCy
    - Deployed using Streamlit
    - Supports multiple entity types
    
    Ideal for NLP demos and learning.
    """
)
