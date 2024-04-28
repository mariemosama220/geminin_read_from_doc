
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


genai.configure(api_key="AIzaSyB9h8ue38uvwiqRh7htpUAPRrZmi6Ei9Ug")

if 'history' not in globals():
    history=[""]



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key="AIzaSyB9h8ue38uvwiqRh7htpUAPRrZmi6Ei9Ug")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    

def get_conversational_chain():
    prompt_template = """
    You are assistant for question-answering tasks on the website of the Higher Institute of Computers and Information Technology in Shorouk City. Your purpose is to assist students in obtaining information about the education system.
    Please answer each question in as much detail as possible, Search well for answer drawing from the provided context. If the question is in Arabic, kindly request the user to provide it in English.
    Try to answer all questions, Provide accurate answers
    Context: \n{context}?\n
    Question: \n{question}\n
    Chat History : {history}
    Answer: 
"""
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.8,google_api_key='AIzaSyB9h8ue38uvwiqRh7htpUAPRrZmi6Ei9Ug',top_p=0.85)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question" , "history"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='AIzaSyB9h8ue38uvwiqRh7htpUAPRrZmi6Ei9Ug')
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question, "history":history[:2] + history[-2:]}, return_only_outputs=True)
    history.append("Question: " + user_question + ". Answer: " + response['output_text'] )
    print("history: ")
    print(history)
    return response['output_text']

def generate_response(prompt_input):
    answer = user_input(prompt_input)
    return answer

text=""
text_chunks = []
pdf_reader= PdfReader("Reference_PDF_Ministry of Higher Education and Scientific Research.pdf")
for page in pdf_reader.pages:
    text+= page.extract_text()
text_chunks = get_text_chunks(text)
get_vector_store(text_chunks)


#streamlit interface
# Custom image for the app icon and the assistant's avatar
assistant_logo = 'https://www.sha.edu.eg/layout/images/logo.png'

# Configure Streamlit page
st.set_page_config(
    page_title="Higher Institute of Computers and Information Technology in Shorouk City Chatbot",
    page_icon=assistant_logo
)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])



# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)