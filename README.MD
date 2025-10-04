
# CureConnect 🩺🤖

CureConnect is an AI-powered medical assistant that combines Large Language Models (LLMs) with retrieval techniques to deliver reliable, context-aware answers.
Built with LLaMA-2, LangChain, FAISS, and Chainlit, it provides a knowledge-driven conversational experience for trustworthy medical support.

# 🚀 Features

● Context-aware medical conversations powered by LLMs.

● Retrieval-Augmented Generation (RAG) for precise, evidence-based answers.

● Combines vector search (FAISS) with LLaMA-2 for knowledge-grounded responses.

● Interactive web interface with Chainlit.

# Project Structure 📂
<img width="909" height="286" alt="Image" src="https://github.com/user-attachments/assets/525571b2-4b0f-4cd7-bb35-cf23e8da1287" />
# Setup Instructions ⚙️ 

1. Clone the Repository

```bash
git clone https://github.com/yuva396/CureConnect.git
cd CureConnect
```
2. Create & Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate 
```
3. Install Requirements
```bash
pip install -r requirements.txt 
```
4. Build the Knowledge Base 
```bash
python ingest.py

```
5. Run the Bot
```bash
chainlit run model.py -w

``
