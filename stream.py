import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

# FAISS DB path
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Custom prompt
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

@st.cache_resource(show_spinner="Loading model and database...")
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    prompt = set_custom_prompt()
    chain = retrieval_qa_chain(llm, prompt, db)
    return chain

# ----------------- Streamlit GUI ---------------------

st.set_page_config(page_title="DiabetesBot", layout="wide")

st.markdown("""
    <style>
        .chat-bubble {
            background-color: #f1f1f1;
            padding: 10px 15px;
            border-radius: 10px;
            margin: 5px 0;
        }
        .user-bubble {
            background-color: #DCF8C6;
            text-align: right;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 DiabetesBot - Chat with Medical Intelligence")

# Sidebar for instructions
with st.sidebar:
    st.header("About")
    st.markdown("""
    This is an AI-powered chatbot designed to help users with questions related to **diabetes**.
    It uses a locally loaded language model and a vector database for contextual answers.
    """)
    st.info("Feel free to ask any question related to diabetes!")

# Load the chain
chain = qa_bot()

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat Input
user_input = st.chat_input("Ask something about diabetes...")

# Display previous chat
for i, (role, msg) in enumerate(st.session_state.messages):
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(msg)

# When user submits a new message
if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append(("user", user_input))

    # Get assistant response
    try:
        response = chain.invoke({"query": user_input})
        answer = response["result"]
    except Exception as e:
        answer = "Sorry, something went wrong while fetching the response."

    # Show assistant message
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append(("assistant", answer))
