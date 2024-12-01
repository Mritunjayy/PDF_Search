# **PDF Search Application**

This is a Python-based **PDF Search Application** built using **Streamlit**, **PyTorch**, **LLM** and **Sentence Transformers**. It allows users to upload a PDF, extract and chunk its content, create embeddings for efficient text search, and answer user queries about the document using a specified language model.

## **Features**
1. Upload a PDF and extract its content.
2. Split the content into chunks for processing.
3. Create and store embeddings using Sentence Transformers for fast similarity searches.
4. Retrieve and display the most relevant chunks for a user query.
5. Answer user questions based on document context using the **Groq API**.
6. Manage and update API keys securely.

---

## **Technologies Used**
- **Streamlit**: For creating the web-based user interface.
- **PyTorch**: For handling embeddings and similarity computation.
- **Sentence Transformers**: For generating text embeddings.
- **Pyngrok**: To make the Streamlit app publicly accessible.
- **Fitz (PyMuPDF)**: For extracting text from PDF files.
- **Groq API**: For natural language processing tasks.

---

## **Setup Instructions**

### **Prerequisites**
1. Install Python 3.8 or later.
2. Ensure `pip` is available for package installation.
3. Sign up for a Groq API key (if using Groq's language models).

### **Installation**
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>

2. Install the necessary packages
    ```bash
    pip install streamlit sentence-transformers torch pymupdf tqdm pyngrok

3. Set Up the Groq API:
    Save your Groq API key in a file named api.txt in the project directory.

4. Run the application
    ```bash
    streamlit run app.py

5. Open the local URL provided in the terminal (e.g., http://localhost:8501).

### **Usage**
1. Upload a PDF document.
2. The app processes the PDF by:
    . Extracting the text content.
    . Splitting the text into chunks with overlap.
    . Generating and storing embeddings for each chunk.
3. Enter a query to search for relevant chunks in the document.
4. Use the saved or updated Groq API key to generate answers based on the retrieved document context.



