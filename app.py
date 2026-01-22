# Importing required libraries
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
from pdf2text import text_extractor
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Setup LLM Model
gemini_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Setup Embedding Model
embedding_model = HuggingFaceBgeEmbeddings(model_name ='all-MiniLM-L6-v2')

# Lets create the main page
st.title(":orange[RAG-Bot:] :blue[AI Assisted Chatbot using RAG]")
tips='''
Follow the steps to use the application:
* Upload your PDF Document in sidebar.
* Write a query and start the chat.'''
st.text(tips)

# Lets create the sidebar
st.sidebar.markdown('# :red[ UPLOAD YOUR FILE]')
st.sidebar.subheader("Upload PDF file only")
pdf_file = st.sidebar.file_uploader("Upload here",type=["pdf"])

MAX_TURNS = 5

if pdf_file:
    st.sidebar.success("File uploaded successfully")
    
    file_text=text_extractor(pdf_file)

    # Step 1: Chunking
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = splitter.split_text(file_text)
    
    # Step 2: Create the vector database (FAISS)
    vector_store = FAISS.from_texts(chunks,embedding_model)
    retriver = vector_store.as_retriever(search_kwargs={'k':3})
    
    def get_chat_history(exclude_latest=True):
        history = ""
        
        messages = st.session_state.history
        
        # Exclude last user message (current query)
        if exclude_latest and len(messages) > 0:
            messages = messages[:-1]
        
        trimmed_history = st.session_state.history[-(MAX_TURNS * 2):]
        
        for msg in st.session_state.history:
            history += f"{msg['role'].upper()}: {msg['text']}\n"
        return history
    
    def generate_content(query):
        # Step 3: Retrieval (R)
        retrived_docs = retriver.invoke(query)
        context = '\n'.join([d.page_content for d in retrived_docs])
        
        chat_history = get_chat_history(exclude_latest=True)
        
        # Step 4: Augmenting (A)
        augmented_prompt = f'''
        <Role> You are a helpful assistant using Conversational RAG.
        <Goal> Answer the questions asked by the user. Here is the question {query}
        <Context>
        * Here are the documnets retrieved from the vector Database to support the answer which you have to generate: {context}
        * Here is the conversation history {chat_history}
        <Instructions>
        - User conversation history to understand follow-up questions.
        - Answer only from the provided context.
        '''
        
        # Step 5: Generate (G)
        response = model.generate_content(augmented_prompt)
        return response.text
    
    # Lets create ChatBot in order to start the conversation
    # Initialize a chat create history if not created
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Display the History
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.write(f":green[USER:] :blue[{msg['text']}]")
        else:
            st.write(f":orange[BOT:] {msg['text']}")
    
    # Input from the user using streamlit form
    with st.form('Chatbot Form',clear_on_submit=True):
        user_query = st.text_area('Ask Anything')
        send = st.form_submit_button('send')
        
    # Start the conversation and append the output and query in history
    
    if user_query and send:
        st.session_state.history.append({'role':'user',
                                         'text':user_query})
        
        bot_response = generate_content(user_query)
        
        st.session_state.history.append({'role':'bot',
                                         'text':bot_response})
        st.rerun()