import streamlit as st
import fitz
import os
from sentence_transformers import SentenceTransformer, util
import torch
from groq import Groq
from tqdm import tqdm

# File to store the API key locally
API_KEY_FILE = "api.txt"
API_KEY_FILE_UPDATED = "updated_api.txt"

# Function to load the API key from the local file
def load_api_key():
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, "r") as file:
            return file.read().strip()
    return None

# Function to save the API key to the local file
def save_api_key(api_key):
    print(api_key.encode)
    with open(API_KEY_FILE_UPDATED, "w") as file:
        file.write(api_key)

# Initialize Streamlit session state for the API key
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = load_api_key()  # Load from local fi

# Load PDF
def load_pdf(pdf_file_path):
    contents = []
    with fitz.open(pdf_file_path) as doc:
        for page in doc:
            content = page.get_text()
            contents.append("\n" + content)
    return "\n".join(contents)


# Chunk Documents
def chunk_documents(text, chunk_size, overlap):
    """
    Splits the input text into smaller chunks of specified size with optional overlap.
    """
    split_lists = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))  # Ensure we don't exceed text length
        split_lists.append(text[start:end])  # Extract chunk
        start += chunk_size - overlap  # Move start with overlap
    return split_lists

# Store Embeddings
def store_embeddings_as_dict(embeddings_list, text_list):
    return {text: embedding for text, embedding in zip(text_list, embeddings_list)}

def save_embeddings_to_pt(embeddings_dict, filename):
    torch.save(embeddings_dict, filename)

def create_vectorstore(embed_model, chunks):
    chunk_embeddings = []
    embedder = SentenceTransformer(embed_model)
    try:
        embedder = SentenceTransformer(embed_model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
    for chunk in tqdm(chunks, desc="Computing Embeddings..."):
        chunk_embedding = embedder.encode(chunk, convert_to_tensor=True).to("cpu")
        chunk_embeddings.append(chunk_embedding)
    embeddings_dict = store_embeddings_as_dict(chunk_embeddings, chunks)
    save_embeddings_to_pt(embeddings_dict, "embeddings.pt")

# Retrieve Relevant Docs
def retrieve_relevant_docs(embed_model, embeddings_path, query, top_k):
    # embedder = SentenceTransformer(embed_model)
    try:
        embedder = SentenceTransformer(embed_model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

    embeddings_dict = torch.load(embeddings_path)
    chunks = list(embeddings_dict.keys())
    query_encoded = embedder.encode(query, convert_to_tensor=True)
    scores = [util.cos_sim(query_encoded, embeddings_dict[chunk])[0] for chunk in chunks]
    top_k_indices = torch.topk(torch.Tensor(scores), k=min(top_k, len(chunks))).indices.tolist()
    return [chunks[i] for i in top_k_indices]

# Stuff Documents
def stuff_docs(relevant_chunks):
    return "".join([f"Document{i}: {chunk}\n\n" for i, chunk in enumerate(relevant_chunks, start=1)])



def qa(llm_name, context, question, api_key):
    prompt = f"Answer the question based on the given context alone.\\ncontext: {context}\\nquestion: {question}\\nanswer:"
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=llm_name,
    )
    return chat_completion.choices[0].message.content

# Main Streamlit App
st.title("PDF Search App")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
# webpage_url = st.text_input("Or, enter a webpage URL:")


if uploaded_file:
    
    # Save the uploaded file
    pdf_path = f"/{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Processing PDF...")
    docs = load_pdf(pdf_path)
    st.success("PDF loaded successfully!")
    
    # Chunk the document
    chunk_size = 500
    overlap = 50
    chunks = chunk_documents(docs, chunk_size, overlap)
    st.write(f"Document split into {len(chunks)} chunks.")
    
    # Create vectorstore
    embed_model = "all-MiniLM-L6-v2"
    create_vectorstore(embed_model, chunks)
    st.success("Embeddings created and stored!")

    # Query the document
    query = st.text_input("Ask a question about the document:")
    if query:
        top_k_docs = retrieve_relevant_docs("all-MiniLM-L6-v2", "embeddings.pt", query, 3)
        stuffed_doc = stuff_docs(top_k_docs)

        if st.session_state.groq_api_key:
            st.info(f"Using saved API Key: {st.session_state.groq_api_key[:4]}****")
        else:
            st.warning("No saved API Key found. Please update the API Key.")

        # Display the buttons for user options
        col1, col2 = st.columns(2)


        if "show_text_input" not in st.session_state:
            st.session_state.show_text_input = False

        # Button to use the saved API key
        if col1.button("Use Saved API Key"):
            if st.session_state.groq_api_key:
                st.success("Proceeding with the saved API Key...")
                api_key_for_use = st.session_state.groq_api_key
                if st.session_state.groq_api_key:
                    answer = qa("llama3-70b-8192", stuffed_doc, query, st.session_state.groq_api_key)
                    st.success(f"Answer: {answer}")
            else:
                st.error("No API Key found. Please update the API Key.")

        # Button to update the API key
        if col2.button("Update API Key"):
            st.session_state.show_text_input = True 

        if st.session_state.show_text_input:
            new_api_key = st.text_input("Enter new Groq API Key:", type="password")
            if st.button("Submit New API Key"):
                if new_api_key:
                    save_api_key(new_api_key)  # Save the new API key to file
                    st.session_state.groq_api_key = new_api_key  # Update session state
                    st.success("API Key updated and saved successfully!")
                    st.session_state.show_text_input = False  # Hide the text input after submission
                    if st.session_state.groq_api_key:
                        answer = qa("llama3-70b-8192", stuffed_doc, query, st.session_state.groq_api_key)
                        print("printing the answer")
                        st.success(f"Answer: {answer}")
                else:
                    st.error("Please enter a valid API Key.")

                if st.session_state.groq_api_key:
                    answer = qa("llama3-70b-8192", stuffed_doc, query, st.session_state.groq_api_key)
                    print("printing the answer")
                    st.success(f"Answer: {answer}")
    # ------------------------------
        
from pyngrok import ngrok

# Set your ngrok auth token
ngrok.set_auth_token("2pTk7C0to9KchpoNZE7sl4t3dLw_2XPZTZJpQAGX8abVpE6NJ")

# Start Streamlit app using ngrok

ngrok.kill()
public_url = ngrok.connect(addr=8501)
print(f"Streamlit app is live at: {public_url}")
