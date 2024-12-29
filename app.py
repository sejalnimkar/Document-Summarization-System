import streamlit as st
import os
from transformers import pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from docx import Document  # To handle .docx files
import base64
import concurrent.futures

# Function to add the background image


def set_background(image_file):
    """Sets a background image to the Streamlit app."""
    img_bytes = image_file.read()  # Read the image as bytes
    img_base64 = base64.b64encode(img_bytes).decode(
        'utf-8')  # Base64 encode the image
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url('data:image/jpeg;base64,{img_base64}');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            .stFileUploader div {{
                color: #00008B !important;
            }}
            body, .stApp {{
                color: #00008B;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Streamlit Interface
st.set_page_config(
    page_title="Intelligent Document Summarization and Q&A System", layout="wide")

# Apply a background image
background_image_path = r"C:\Users\danim\Downloads\network-mesh-wire-digital-technology-background\17973908.jpg"
set_background(open(background_image_path, "rb"))


def is_sensitive_document(text):
    """Detects if a document contains sensitive content using enhanced regular expressions."""
    sensitive_patterns = [
        r"\b\d{9}\b",  # Generic 9-digit numbers (e.g., common in passports)
        # Passport numbers with 1-2 letters followed by 7 digits
        r"\b[A-Z]{1,2}\d{7}\b",
        # Example: UK passport format (2 letters + 8 digits)
        r"\b[A-Z]{2}\d{8}\b",
        r"\d{3}-\d{2}-\d{4}",  # Social Security Number (US format)
        r"\b\d{16}\b",  # Credit card numbers (basic 16-digit format)
        r"\b\d{12,19}\b",  # Generic bank card numbers (12-19 digits)
        # Bank account number (example format)
        r"\b[A-Z0-9]{1,20} \d{4}-\d{6}-\d{7}\b",
    ]

    for pattern in sensitive_patterns:
        if re.search(pattern, text):
            return True
    return False


def load_and_split_pdf(file):
    """Load and split PDF files."""
    temp_file_path = f"temp_{file.name}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file.read())
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    os.remove(temp_file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=300)
    return text_splitter.split_documents(documents)


def load_txt(file):
    """Load text from a .txt file."""
    return file.read().decode("utf-8")


def load_docx(file):
    """Load text from a .docx file."""
    doc = Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


# Initialize session state for summary, uploaded file name, and question
if "summary" not in st.session_state:
    st.session_state.summary = ""

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "question" not in st.session_state:
    st.session_state.question = ""

# Initialize the summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Streamlit Interface
st.title("Intelligent Document Summarization and Q&A System")
uploaded_file = st.file_uploader(
    "Upload a PDF, Text, or Word Document", type=["pdf", "txt", "docx"])

if uploaded_file:
    # Reset the summary and question if a new file is uploaded
    if st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.summary = ""
        st.session_state.question = ""
        st.session_state.uploaded_file_name = uploaded_file.name

    with st.spinner("Processing the uploaded document..."):
        try:
            # Handle file based on type
            if uploaded_file.type == "application/pdf":
                split_documents = load_and_split_pdf(uploaded_file)
                full_text = " ".join(
                    [doc.page_content for doc in split_documents])
            elif uploaded_file.type == "text/plain":
                full_text = load_txt(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                full_text = load_docx(uploaded_file)
            else:
                st.error("Unsupported file type.")
                st.stop()

            # Check for sensitive content
            if is_sensitive_document(full_text):
                st.error(
                    "This document contains sensitive information and cannot be processed.")
            else:
                st.success(
                    "No sensitive content detected. Proceeding with summarization and Q&A.")

                # Display the summary if it exists
                if st.session_state.summary:
                    st.markdown("### Summary:")
                    st.text_area(
                        "Generated Summary", st.session_state.summary, height=200, disabled=True)

                # Summarization Section
                if st.button("Summarize Document"):
                    with st.spinner("Generating summary..."):
                        # Function to generate detailed summary with dynamic max_length and chunk size
                        def generate_detailed_summary(text, chunk_size=1024, max_length=300, min_length=75, max_chunks=5):
                            # Split document into chunks
                            chunks = [text[i:i + chunk_size]
                                      for i in range(0, len(text), chunk_size)]

                            # Limit the number of chunks for faster processing
                            chunks = chunks[:max_chunks]

                            # Define a helper function for summarizing a chunk
                            def summarize_chunk(chunk):
                                # Number of words in chunk
                                input_length = len(chunk.split())
                                dynamic_max_length = min(
                                    max_length, max(100, input_length * 2))
                                chunk_summary = summarizer(
                                    chunk, max_length=dynamic_max_length, min_length=min_length, do_sample=False)
                                return chunk_summary[0]["summary_text"]

                            # Use parallel processing to speed up summarization
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                summaries = list(executor.map(
                                    summarize_chunk, chunks))

                            # Return combined summary
                            return " ".join(summaries)

                        # Generate the summary and display it
                        summary = generate_detailed_summary(full_text)
                        st.session_state.summary = summary  # Store the summary in session state
                        st.text_area(
                            "Generated Summary", summary, height=200, disabled=True)

                # Q&A Section
                st.markdown("### Ask a Question About the Document")
                st.session_state.question = st.text_input(
                    "Enter your question:", value=st.session_state.question)
                if st.button("Get Answer") and st.session_state.question:
                    with st.spinner("Searching for the answer..."):
                        qa_model = pipeline(
                            "question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
                        answer = qa_model(
                            question=st.session_state.question, context=full_text)
                        st.markdown("### Answer:")
                        st.write(answer["answer"])
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
