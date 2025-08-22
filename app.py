pip install streamlit pandas openai chromadb transformers openpyxl
import streamlit as st
import pandas as pd
import openai
from transformers import AutoTokenizer, AutoModel
import chromadb
import numpy as np

# ----- UI -----
st.title("RAG Chatbot for Tabular Excel Data")
openai_api_key = st.text_input("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", type='password')

uploaded_file = st.file_uploader("Upload your Excel data file", type=["xlsx"])
if uploaded_file and openai_api_key:
    df = pd.read_excel(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # ----- Row Serialization -----
    serialized_rows = []
    for idx, row in df.iterrows():
        serialized = "; ".join(f"{col}: {row[col]}" for col in df.columns)
        serialized_rows.append(serialized)

    # ----- Embedding Model Setup -----
    st.info("Generating embeddings for each row (first time is slower)...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    def get_embedding(text):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    embeddings = np.array([get_embedding(text) for text in serialized_rows])
    
    # ----- Create Chroma Vector DB -----
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("excel_table")
    for idx, emb in enumerate(embeddings):
        collection.add(
            embeddings=[emb.tolist()],
            ids=[str(idx)],
            documents=[serialized_rows[idx]]
        )
    st.success("Embeddings stored in vector database.")

    # ----- Chatbot UI -----
    st.subheader("Ask Questions About Your Data")
    user_query = st.text_input("Your question (e.g., 'Monthly active users in Q2?')")

    if user_query and st.button("Ask"):
        # Get embedding for user query
        q_emb = get_embedding(user_query)
        # Vector search for top relevant rows
        results = collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=5
        )
        retrieved_context = "\n".join(results['documents'])
        st.info("Retrieved relevant data rows:")

        st.text_area("Retrieved Context for Answering", value=retrieved_context, height=150)
        # Compose prompt for LLM
        prompt = (
            f"Here are some rows from a table:\n{retrieved_context}\n\n"
            f"Answer the following question using ONLY the data above. "
            f"Provide clear, concise reasoning in your answer.\n"
            f"Question: {user_query}\n"
            "Answer:"
        )
        try:
            openai.api_key = openai_api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analyst for Excel tables."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300
            )
            summary = response.choices[0].message.content
            st.markdown("### Answer")
            st.markdown(summary)
        except Exception as e:
            st.error(f"API Error: {e}")

