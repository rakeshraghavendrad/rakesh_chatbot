import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Backend URL
BACKEND_URL = "http://localhost:5000/ask"

# Streamlit app configuration
st.set_page_config(
    page_title="HR Insights Assistant",
    layout="centered",
    initial_sidebar_state="expanded"
)

# App title
st.title("HR Insights Assistant")
st.markdown(
    "This tool answers your questions based on HR documents and graph insights. "
    "You can also request flowcharts or tables if relevant to the context."
)

# User input
question = st.text_input("Ask your question:", placeholder="Type your question here...")

# Submit button
if st.button("Submit"):
    if not question.strip():
        st.error("Please enter a question before submitting.")
    else:
        # Send question to backend
        with st.spinner("Fetching response..."):
            try:
                response = requests.post(BACKEND_URL, json={"question": question})

                if response.status_code == 200:
                    data = response.json()

                    # Display response text
                    st.subheader("Response")
                    st.markdown(data.get("response_text", "No response generated."))

                    # Display table if available
                    tables_html = data.get("tables_html", "")
                    if tables_html:
                        st.subheader("Tables")
                        st.markdown(tables_html, unsafe_allow_html=True)

                    # Display flowchart if available
                    flowchart_path = data.get("flowchart_path", None)
                    if flowchart_path:
                        st.subheader("Flowchart")
                        response = requests.get(f"http://localhost:5000/{flowchart_path}")
                        if response.status_code == 200:
                            image = Image.open(BytesIO(response.content))
                            st.image(image, caption="Generated Flowchart", use_column_width=True)
                        else:
                            st.error("Failed to load flowchart.")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the backend: {e}")

# Footer
st.markdown("---")
st.caption("Powered by Streamlit & Flask")
