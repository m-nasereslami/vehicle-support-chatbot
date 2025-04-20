# 🚗 AI-Powered Vehicle Support Chatbot

## Overview
This repository contains an AI-powered chatbot that provides vehicle support by answering queries from a **vehicle manual** and recognizing car parts from images.

The chatbot utilizes:
- **Retrieval-Augmented Generation (RAG)** for intelligent question-answering.
- **FAISS Vector Store** for fast document retrieval.
- **Hugging Face Falcon-7B Instruct** for generating accurate responses.
- **YOLOv8** for detecting car parts in images.
- **FastAPI** for backend API support.
- **Streamlit** for a user-friendly chatbot UI.

---

## Files in this Repository
- **`app.py`** → Main script containing the chatbot, AI model, and API.
- **`vehicle_manual.pdf`** → The vehicle manual used for reference.

---

## Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/m-nasereslami/vehicle-support-chatbot.git
cd vehicle-support-chatbot
```

### **2️⃣ Create a Virtual Environment**
```bash
conda create --name vehicle_ai python=3.9
conda activate vehicle_ai
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Set Up the Hugging Face API Key**
```bash
export HF_API_KEY="your_huggingface_api_key"
```

---

## Usage

### **Run the Chatbot & API**
```bash
streamlit run app.py --server.headless false
```

### **Access the Chatbot UI**
- Open **Streamlit** in your browser:
  ```
  http://localhost:8501
  ```

### **Use the API**
- Send a POST request to FastAPI:
  ```bash
  curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"query": "When should I change engine oil?"}'
  ```

---

## Features
 **Vehicle Manual QA** → Answers vehicle-related questions.

 **Car Part Recognition** → Detects car parts from uploaded images.

 **FastAPI Backend** → API for integration with other applications.
 
 **Streamlit UI** → Easy-to-use chatbot interface.

---




