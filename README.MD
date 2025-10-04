
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
# Bot in Action
CureConnect is an AI-powered health assistant designed to make medical information more accessible and reliable. It allows users to ask natural-language questions about symptoms, diseases, and medications, and receive responses generated from trusted medical knowledge sources.

<img width="1909" height="868" alt="Image" src="https://github.com/user-attachments/assets/92a5e280-fd35-45d3-8275-e94e3c9e91ed" />

<img width="1919" height="877" alt="Image" src="https://github.com/user-attachments/assets/69cbcafe-0828-42ae-9673-82b2bf6deef6" />

<img width="1919" height="878" alt="Image" src="https://github.com/user-attachments/assets/a99ffe4d-1df8-462e-8060-1d8640f07c6d" />
