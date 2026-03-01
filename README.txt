# 📄 Private-RAG: Local Document Intelligence
### FastAPI + Streamlit + Qwen3-30B + Docker



This project is a high-performance, **locally-hosted** Retrieval-Augmented Generation (RAG) pipeline. It allows you to chat with your private PDF documents without your data ever leaving your machine. By decoupling the logic (FastAPI) from the interface (Streamlit), the system remains responsive even when handling heavy 30B parameter models.

---

## ✨ Features
* **Fully Local:** No OpenAI API keys required. Your data stays on your hardware.
* **Decoupled Architecture:** FastAPI backend handles the heavy lifting; Streamlit provides a sleek UI.
* **High-End LLM Support:** Optimized for **Qwen3-30B-A3B** (AWQ 4-bit) for near-GPT-4 level reasoning locally.
* **Vector Search:** Powered by **FAISS** and **Nomic-Embed-Text** for lightning-fast document retrieval.
* **Dockerized:** One command to spin up the entire environment with GPU passthrough.

---

## 💻 Hardware Requirements
To run a **30B model** comfortably, you generally need:
* **GPU:** NVIDIA GPU with **24GB+ VRAM** (e.g., RTX 3090, 4090, or A6000).
* **RAM:** 32GB+ System RAM.
* **Drivers:** [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed for Docker GPU support.

---

## 📂 Project Structure
```text
.
├── backend.py            # FastAPI Server (LLM & Vector Store logic)
├── frontend.py           # Streamlit Web Interface
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container definition
├── docker-compose.yml    # Multi-container orchestration
└── us_census/            # Directory for your PDF documents


🚀 Quick Start (with Docker)
Clone the Repository
git clone [https://github.com/your-repo/local-rag.git](https://github.com/your-repo/local-rag.git)
cd local-rag

Add Your Documents
Drop your .pdf files into the us_census/ folder.

Launch the Stack
docker-compose up --build

Access the App

Frontend (UI): http://localhost:8501

Backend (API Docs): http://localhost:8000/docs

Method,Endpoint,Description
POST,/ingest,Scans the PDF folder and rebuilds the FAISS index.
POST,/ask,Takes a JSON question and returns a RAG-based answer.