import pickle
import pandas as pd
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load Diabetes Model
with open('models/trained_model.pkl', 'rb') as f:
    diabetes_model = pickle.load(f)

REQUIRED_FEATURES = [
    'Age', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL',
    'BMI', 'Glucose', 'BloodPressure', 'Insulin',
    'DiabetesPedigreeFunction'
]

class FeatureCollector:
    def __init__(self):
        self.collected_features = {}
        self.current_feature_idx = 0
    
    def get_next_question(self) -> str:
        if self.current_feature_idx >= len(REQUIRED_FEATURES):
            return None
        
        feature = REQUIRED_FEATURES[self.current_feature_idx]
        questions = {
            'Age': "What's your age?",
            'HbA1c': "What's your HbA1c level? (%)",
            'Chol': "What's your total cholesterol level? (mg/dL)",
            'TG': "What's your triglyceride level? (mg/dL)",
            'HDL': "What's your HDL cholesterol level? (mg/dL)",
            'LDL': "What's your LDL cholesterol level? (mg/dL)",
            'VLDL': "What's your VLDL cholesterol level? (mg/dL)",
            'BMI': "What's your BMI?",
            'Glucose': "What's your fasting glucose level? (mg/dL)",
            'BloodPressure': "What's your blood pressure? (mmHg)",
            'Insulin': "What's your insulin level? (µU/mL)",
            'DiabetesPedigreeFunction': "What's your diabetes pedigree function value?"
        }
        return questions[feature]
    
    def store_feature(self, value: str):
        feature = REQUIRED_FEATURES[self.current_feature_idx]
        try:
            self.collected_features[feature] = float(value)
            self.current_feature_idx += 1
        except ValueError:
            return f"Please enter a valid number for {feature}"
        return None
    
    def ready_for_prediction(self) -> bool:
        return len(self.collected_features) == len(REQUIRED_FEATURES)
    
    def make_prediction(self):
        input_data = pd.DataFrame([self.collected_features])
        prediction = diabetes_model.predict(input_data)[0]
        proba = diabetes_model.predict_proba(input_data)[0][1]
        return prediction, proba

def setup_rag():
    # Load documents
    loader = PyPDFLoader("Diabets.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Setup embeddings (local)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Use local Ollama LLM
    llm = Ollama(
        model="tinyllama",  # or "mistral"
        temperature=0.5
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

class DiabetesChatbot:
    def __init__(self):
        self.feature_collector = FeatureCollector()
        self.qa = setup_rag()
        self.state = "IDLE"
    
    def respond(self, user_input: str) -> str:
        if "diabetes risk" in user_input.lower():
            self.state = "COLLECTING_FEATURES"
            return self.feature_collector.get_next_question()
        
        if self.state == "COLLECTING_FEATURES":
            error = self.feature_collector.store_feature(user_input)
            if error:
                return error
            
            if self.feature_collector.ready_for_prediction():
                prediction, proba = self.feature_collector.make_prediction()
                self.state = "IDLE"
                return self._format_prediction(prediction, proba)
            else:
                return self.feature_collector.get_next_question()
        else:
            return self.qa.invoke({"query": user_input})["result"]
    
    def _format_prediction(self, prediction: int, probability: float) -> str:
        risk = "High risk" if prediction == 1 else "Low risk"
        percentage = f"{probability*100:.1f}%"
        
        recommendations = {
            1: "Please consult a doctor soon. Consider:\n"
               "- Reducing sugar intake\n"
               "- Regular exercise\n"
               "- Frequent monitoring",
            0: "Maintain healthy habits:\n"
               "- Balanced diet\n"
               "- Annual checkups\n"
               "- Moderate exercise"
        }
        
        return (
            f"Diabetes Risk Assessment:\n"
            f"Risk Level: {risk} ({percentage})\n\n"
            f"Recommendations:\n{recommendations[prediction]}"
        )

if __name__ == "__main__":
    bot = DiabetesChatbot()
    print("Bot: Hello! How can I help you today? (Type 'quit' to exit)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        response = bot.respond(user_input)
        print(f"Bot: {response}")