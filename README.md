# Intelligent Document Summarization and Q&A System

This is a Streamlit-based application for document summarization and question-answering. It supports processing of **PDF**, **Word (.docx)**, and **Text (.txt)** files while checking for sensitive information. The app provides detailed summaries and allows users to ask questions based on the document content.

## Features
- **Upload and Process Documents**: Supports PDF, TXT, and DOCX files.
- **Summarization**: Generates concise summaries using a pre-trained model.
- **Question-Answering**: Ask questions and receive context-aware answers from the document.
- **Sensitive Content Detection**: Identifies and restricts processing of sensitive information (e.g., SSNs, passport numbers, etc.).
- **Background Customization**: Customizable background for an enhanced user interface.

## Technologies Used
- **Streamlit**: For creating the web-based application.
- **Hugging Face Transformers**: Provides pre-trained models for summarization and Q&A.
- **LangChain**: Helps with PDF text extraction and splitting.
- **Python-docx**: For reading `.docx` files.
- **Regex**: For detecting sensitive content.
