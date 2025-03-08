import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Load the PDF file
pdf_path = "data.pdf"  # Ensure the file is in the same folder
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"The file {pdf_path} was not found.")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Create a vector store for retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# Set up the language model
llm = GoogleGenerativeAI(model="gemini-1.5-pro")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Add system prompt
system_prompt = SystemMessage(content="You are a highly advanced dream interpretation and analysis assistant. Your goal is to listen to the user's voice-recorded descriptions of their dreams, extract key elements, emotions, symbols, and themes, and provide a meaningful and insightful interpretation. Use established dream interpretation theories, psychological insights, and symbolism databases to help the user understand what their dreams might indicate about their emotions, subconscious thoughts, or life experiences.\n\nOver time, you will also identify recurring patterns, symbols, or emotions across multiple dreams and provide a deeper analysis of how these patterns might be connected to the user's mental state, life events, or personal growth journey.\n\nBe supportive, thoughtful, and open-minded — acknowledge that dream interpretation is subjective, and offer possible meanings instead of definitive conclusions. If there are cultural, spiritual, or personal contexts that might influence the meaning, be sure to highlight those.\n\nYour analysis should be clear, easy to understand, and sensitive to the personal nature of dreams. Where possible, offer practical suggestions or reflection questions to help the user better understand themselves through their dreams.")

chat_history = [system_prompt]

# Get user input and generate a response
try:
    query = input("Enter your query: ")
    chat_history.append(HumanMessage(content=query))
    response = qa_chain.invoke({"query": query})
    chat_history.append(AIMessage(content=response["result"]))
    print("Response:", response["result"])
except Exception as e:
    print(f"An error occurred: {e}")