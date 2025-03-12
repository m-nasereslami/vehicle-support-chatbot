import os
import fitz  
import torch
import cv2
import streamlit as st
import uvicorn
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from ultralytics import YOLO
from huggingface_hub import InferenceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain.chains import RetrievalQA

# API Key 
HF_API_KEY = os.getenv("HF_API_KEY")

# ========== STEP 1: Extract Text from PDFs and Store in FAISS ==========

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF document."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Error: The file '{pdf_path}' was not found.")
    
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)

# Load & process document
pdf_path = "vehicle_manual.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

# Split text into smaller chunks for vector search
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
documents = splitter.split_text(pdf_text)

# Create FAISS vector store using Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(documents, embeddings)
vector_store.save_local("faiss_index")

# ========== STEP 2: Load FAISS and LLM for RAG Chatbot ==========

vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever()

# Hugging Face Inference API Client
client = InferenceClient(model="tiiuae/falcon-7b-instruct", token=HF_API_KEY)

def generate_response(query: str) -> str:
    """Retrieves relevant information and generates a response using an LLM."""
    retrieved_docs = retriever.get_relevant_documents(query)
    if not retrieved_docs:
        return "I don't know. The manual does not contain this information."
    
    context = " ".join([doc.page_content for doc in retrieved_docs])
    prompt = (
        "Use ONLY the following context to answer the question. "
        "DO NOT make up any information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    
    return client.text_generation(prompt, max_new_tokens=50, temperature=0.1).strip()

# ========== STEP 3: Car Part Recognition with YOLOv8 ==========

def detect_car_parts(image_path: str) -> list:
    """Detects car parts in an image using YOLOv8."""
    model = YOLO("yolov8n.pt")  # Load pre-trained YOLO model
    results = model(image_path)
    return [result.names[int(box.cls[0])] for result in results for box in result.boxes]

# ========== STEP 4: Streamlit Chatbot UI ==========

def chatbot_ui():
    """Streamlit UI for vehicle support chatbot."""
    st.title("🚗 AI-Powered Vehicle Support Chatbot")
    query = st.text_input("Ask me anything about your vehicle:")
    uploaded_file = st.file_uploader("Upload a car part image", type=["jpg", "png"])
    
    if st.button("Get Answer"):
        if uploaded_file:
            image = Image.open(uploaded_file)
            image_path = "uploaded_image.jpg"
            image.save(image_path)
            detected_parts = detect_car_parts(image_path)
            st.write(f"Detected parts: {detected_parts}")
        
        if query:
            response = generate_response(query)
            st.write(response)

# ========== STEP 5: FastAPI Backend ==========

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    """API endpoint for answering vehicle-related queries."""
    response = generate_response(request.query)
    return {"answer": response}

if __name__ == "__main__":
    # Run Streamlit UI and FastAPI simultaneously
    chatbot_ui()
    uvicorn.run(app, host="0.0.0.0", port=8000)
