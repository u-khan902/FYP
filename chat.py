import streamlit as st
import numpy as np
import pickle
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

# === Load Diabetes Prediction Model ===
@st.cache_resource
def load_prediction_model():
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# === Load Chatbot components ===
DB_FAISS_PATH = 'vectorstore/db_faiss'

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

def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    prompt = set_custom_prompt()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

qa_chain = qa_bot()

# === Streamlit UI ===
st.set_page_config(page_title="Diabetes App", layout="wide")
st.title("🩺 DiabetesBot: Prediction & Chat")

# Sidebar navigation
menu = st.sidebar.radio("Choose an Option", ("Diabetes Prediction", "Chatbot"))

if menu == "Diabetes Prediction":
    st.header("🔬 Diabetes Prediction")
    
    # Input fields
    age = st.number_input("Age (years) (18-80)", min_value=0)
    hba1c = st.number_input("HbA1c (%) (3-15)", min_value=0.0)
    chol = st.number_input("Cholesterol (0.9- 10.3) mmol/L (millimoles per liter)", min_value=0.0)
    tg = st.number_input("Triglycerides (0.3 - 13.8) mmol/L (millimoles per liter)", min_value=0.0)
    hdl = st.number_input("HDL (0.2 - 9.9) mmol/L (millimoles per liter)", min_value=0.0)
    ldl = st.number_input("LDL (0.3 - 9.9) mmol/L (millimoles per liter)", min_value=0.0)
    vldl = st.number_input("VLDL (0.2 - 10.3) mmol/L (millimoles per liter)", min_value=0.0)
    bmi = st.number_input("BMI (kg/m²) (15-50)", min_value=0.0)
    glucose = st.number_input("Glucose (mg/dL) (50-500)", min_value=0.0)
    bp = st.number_input("Blood Pressure (mmHg) (50-200)", min_value=0.0)
    insulin = st.number_input("Insulin (μIU/mL) (0-1000)", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function (0-2)", min_value=0.0)

    if st.button("Predict"):
        model = load_prediction_model()
        input_data = np.array([[age, hba1c, chol, tg, hdl, ldl, vldl, bmi, glucose, bp, insulin, dpf]])
        prediction = int(model.predict(input_data)[0])  # 🔧 Fixed: convert array to scalar
        
        result_label = {
            0: "Non-Diabetic",
            1: "Prediabetic",
            2: "Diabetic"
        }

        st.success(f"🩺 Prediction: **{result_label[prediction]}**")

elif menu == "Chatbot":
    st.header("🤖 Diabetes Chatbot")

    # Initialize session state to store chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat input at bottom
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask something related to Diabetes:", key="user_input")
        submit = st.form_submit_button("Send")

    if submit and user_input:
        with st.spinner("Generating response..."):
            response = qa_chain.run(user_input)
        # Save both question and answer
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    # Display conversation history
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**🧑 You:** {message}")
        else:
            st.markdown(f"**🤖 Bot:** {message}")

