import streamlit as st
import requests
import pandas as pd

# FastAPI Backend URL
API_URL = "http://localhost:8000/predict"

# Streamlit UI
st.title("📝 Named Entity Recognition (NER) with BERT")
st.write("Enter a sentence, and the model will detect named entities.")

# User input text box
user_input = st.text_area("Enter text here:", "")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Call FastAPI backend
        response = requests.post(API_URL, json={"text": user_input})
        
        if response.status_code == 200:
            result = response.json()
            tokens = result["tokens"]
            entities = result["entities"]

            # Convert to DataFrame
            df = pd.DataFrame({"Word": tokens, "Entity": entities})

            # Increase column width using Streamlit's table styling
            st.write("### Named Entities Detected")
            st.markdown(
                """
                <style>
                .dataframe th, .dataframe td {
                    padding: 15px;
                    text-align: left;
                    font-size: 18px;
                }
                </style>
                """, unsafe_allow_html=True
            )
            
            # Display table
            st.dataframe(df, width=600, height=300)

        else:
            st.error("Error: Could not connect to the backend API.")