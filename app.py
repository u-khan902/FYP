from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    print("Setting custom prompt...")
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    print("Creating RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Loading the model
def load_llm():
    print("Loading language model...")
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# QA Model Function
def qa_bot():
    print("Initializing QA bot...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    print("Loading FAISS database...")
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    except Exception as e:
        print(f"Error loading FAISS database: {e}")
        return None
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# Chainlit code
@cl.on_chat_start
async def start():
    print("Chat started.")
    chain = qa_bot()
    if chain is None:
        print("Failed to initialize QA bot.")
        return
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to DiabetesBot. How may I help you?"
    await msg.update()
    cl.user_session.set("chain", chain)
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if chain is None:
        print("QA bot not initialized.")
        await cl.Message(content="Failed to initialize QA bot. Please try again later.").send()
        return
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    try:
        res = await chain.ainvoke(message.content, callbacks=[cb])  # Use ainvoke instead of acall
        answer = res["result"]
        await cl.Message(content=answer).send()  # Send only the answer
    except Exception as e:
        print(f"Error during query handling: {e}")
        await cl.Message(content="An error occurred while processing your query.").send()
if __name__ == "__main__":
    # Start Chainlit bot
    cl.run()