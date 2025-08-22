import streamlit as st
import pandas as pd
import openai

# -----------------------------
# 1. Upload Excel
# -----------------------------
st.title("📊 Chat with Your Data (Excel KPIs)")

uploaded_file = st.file_uploader("Upload your Excel/CSV file", type=["xlsx", "csv"])
if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ Data Loaded Successfully")
    st.dataframe(df.head())

    # -----------------------------
    # 2. Select target column
    # -----------------------------
    st.sidebar.header("Settings")
    date_col = st.sidebar.selectbox("Select Date Column", df.columns, index=0)
    user_col = st.sidebar.selectbox("Select User/ID Column", df.columns, index=1)

    # -----------------------------
    # 3. Ask questions
    # -----------------------------
    st.subheader("Ask about your KPIs")
    question = st.text_input("Example: 'What is MAU?', 'Show me QTD sales'")

    if question:
        # Prepare system prompt for LLM
        prompt = f"""
        You are a data analyst. I have this dataframe with columns: {list(df.columns)}.
        Answer the user query using pandas. Query: {question}.
        """

        with st.spinner("Thinking..."):
            # Call OpenAI (or other LLM)
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=200
            )

        answer = response.choices[0].message["content"]
        st.write("💡 Answer:", answer)
