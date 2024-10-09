# Pdf-Chatter

# PDF Question Answering Application with RAG Architecture

This application allows users to upload a PDF document and ask questions based on its content. It uses **Retrieval-Augmented Generation (RAG)** to first retrieve relevant parts of the document and then generate answers using a language model. The app is built using the `langchain`, `transformers`, and `HuggingFace` libraries, leveraging powerful models for both embedding and natural language generation.

## Features

- **PDF Upload and Processing**: Upload a PDF, which is then processed and split into manageable chunks for better retrieval.
- **Contextual Question Answering**: Ask any question based on the PDF content. The app retrieves the relevant parts of the document and generates answers accordingly.
- **Dynamic Responses**: The system uses state-of-the-art models from Hugging Face to ensure accurate and context-aware answers.

## How it Works

1. **PDF Upload**: The PDF is loaded and split into chunks using the `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
2. **Embedding**: The text chunks are converted into embeddings using `HuggingFaceEmbeddings` (specifically, the "all-mpnet-base-v2" model).
3. **Retrieval**: A vector store is created using `Chroma`, and the app retrieves relevant context from the PDF based on the user's question.
4. **Generation**: The Mistral-7B model from Hugging Face is used to generate a natural language response based on the retrieved context.
5. **Output**: The final answer is parsed and displayed to the user.

## Tech Stack

- **PyTorch**: Deep learning framework used for model operations.
- **Transformers**: Hugging Face library for model loading and generation.
- **LangChain**: For handling retrieval and generation pipelines.
- **Chroma**: Vector store used for efficient document retrieval.
- **HuggingFaceHub**: Used to load the Mistral-7B model for question generation.
- **Sentence Transformers**: For embedding the text chunks from the PDF.

## Requirements

To run the application, you need the following packages installed:

```txt
torch
transformers
numpy
langchain
langchain_community
langchain-chroma
sentence_transformers
pypdf
