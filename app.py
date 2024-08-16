import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from prompts import *
import os


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all details.If answer
    is not provided in the context just say,"answer is not available in the context",don't provide the wrong answer \n
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """
    model =ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.7)
    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question":user_question}
        , return_only_outputs=True)

    return response



def main():
    # st.header("Let's Chat with PDF")
    st.markdown("""
    <style>
        .st-emotion-cache-1c7y2kd {
            flex-direction: row-reverse;
            text-align: right;
            background-color: rgba(255, 255, 255);
        }
                
        .st-emotion-cache-janbn0 {
            flex-direction: row-reverse;
            text-align: right;
            background-color: rgb(14 17 23);
        }
        .stButton > button {
            background-color: #4B9CD3; 
            color: white; /* White text */
            border: none; 
            padding: 10px 30px; 
            text-align: center; 
            text-decoration: none; 
            display: inline-block; 
            font-size: 16px; 
            margin: 4px 2px; 
            cursor: pointer; 
            border-radius: 8px; 
            transition: background-color 0.3s; 
        }

        .stButton > button:hover {
            background-color: #007FFF;
            color: white;
        }
    </style>
    """,unsafe_allow_html=True)

    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False

    #### Display Chat history messages 
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]
    # display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = None
    button_name = None

    with st.sidebar:
        st.title("Upload your PDF Files")
        st.image("https://cdn-icons-png.flaticon.com/512/4147/4147103.png", width=100)
        pdf_docs = st.file_uploader("Upload file")
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.pdf_uploaded = True
                    st.success("PDF Uploaded!! Ask Questions")
            else:
                st.error("Please upload at least one PDF file.")

        if st.session_state.pdf_uploaded:
            user_question, button_name = predefined_prompts(user_question, button_name)
        else:
            st.session_state.pdf_uploaded = False
    
    # checking if the prompt given is predefined
    # user_question, button_name = predefined_prompts(user_question,button_name)
    # print(user_question,button_name)

    text_input_question = st.chat_input("Ask a Question")
    if text_input_question:
        user_question = text_input_question
        button_name = user_question
    
    st.session_state.current_prompt= user_question

    if user_question:
        if not pdf_docs:
            with st.sidebar:
                st.error("Please upload at least one PDF file. ")
        else:
            st.session_state.current_prompt = user_question
            st.session_state.current_button = button_name

            if "current_button" in st.session_state and "current_prompt" in st.session_state:
                st.session_state.messages.append({"role": "user", "content": st.session_state.current_button})
                with st.chat_message("user"):
                    st.write(st.session_state.current_button)

                if st.session_state.messages[-1]["role"] != "assistant":
                    with st.chat_message("assistant"):
                        with st.spinner("Analysing..."):
                            # call the RAG Chain
                            response = user_input(st.session_state.current_prompt)
                            st.write(response["output_text"])
                    message = {"role": "assistant", "content": response["output_text"]}
                    st.session_state.messages.append(message)

    
if __name__ == "__main__":
    main()



