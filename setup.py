from setuptools import setup, find_packages

setup(
    name="ner-streamlit-app",
    version="1.0.0",
    description="Named Entity Recognition app built with spaCy and Streamlit",
    author="Yash V",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.31.0",
        "spacy==3.7.2",
    ],
    include_package_data=True,
)
