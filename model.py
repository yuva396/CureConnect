from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

# --- Updated Prompt: Prioritize Most Likely Condition ---
custom_prompt_template = """
Use the following information to answer the user's question.
Only consider the context provided. Ignore anything not mentioned in the context.
Give the most likely condition first. Mention rare causes only if clearly supported.
If you don't know the answer based on the context, say 'I don't know.'

Context: {context}
Question: {question}

Helpful answer:

"""

def set_custom_prompt():
    """Prompt template for QA retrieval"""
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# --- Retrieval QA Chain ---
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# --- Load LLM ---
def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# --- QA Bot ---
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    # Allow deserialization (since PDF is your own)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# --- Output function ---
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# --- Chainlit Handlers ---
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    res = await chain.ainvoke({"query": message.content}, callbacks=[cb])
    answer = res.get("result") or res.get("output_text") or "No result"
    sources = res.get("source_documents") or []

    await cl.Message(content=answer).send()
