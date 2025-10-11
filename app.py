## here it will also remeber previous context
## RAG Q/A chatbot with PDF Including Chathistory
import os
import streamlit as st
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder  ##MessagesPlaceholder(variable_name="chat_history") is the message placeholder.
#from langchain_huggingface import HuggingFaceEmbeddings              ##It’s a slot that will be filled at runtime with the previous conversation (context messages between user and AI).

from dotenv import load_dotenv  
load_dotenv()

##load the groq api key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
#os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = OllamaEmbeddings(model="gemma:2b") 

##set up streamlit
st.title("Conversational  RAG with PDF  uploads  and chat history")
st.write("upload pdfs and chat with their content")

##input the Groq api key
api_key=st.text_input("Enter your Groq API key :",type="password")

##check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model="llama-3.1-8b-instant")
    ##chat interface
    session_id=st.text_input("Session_ID",value="dafault_session")

    ##statefully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store={}  ##dict.

    uploaded_files=st.file_uploader("Choose a PDF file",type="pdf",accept_multiple_files=True) 
    documents=[]

    ##process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:  
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

                loader=PyPDFLoader(temppdf)


    ##split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=200)    
    splits=text_splitter.split_documents(documents)    
    vectorstore=FAISS.from_documents(splits,embeddings) 
    retriever=vectorstore.as_retriever() 

    contexualized_system_prompt=("""given the chat history and latest user question"""  )

    contexualized_system_prompt=ChatPromptTemplate.from_messages(
      [
          ("system",contexualized_system_prompt),
          MessagesPlaceholder(variable_name="chat_history"),  ##it will be replaced by the chat history at runtime
          ("user","{question}")
      ]
  )

    history_aware_retriever=create_history_aware_retriever(llm=llm,retriever=retriever,system_prompt=contexualized_system_prompt)
    

    ##answer question prompt
    system_prompt=("""You are a helpful AI assistant. Use the following pieces of context to answer the users question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
                   {context}"""
                   )
    
    qa_prompt=ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )

    question_answer_chain=create_stuff_documents_chain(llm=llm,prompt=qa_prompt )
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

    def get_session_history(session_id:str):
            if session_id not in st.session_state.store:
              st.session_state.store[session_id]=ChatMessageHistory()
              return st.session_state.store[session_id]

    conversational_rag_chain=RunnableWithMessageHistory(
    rag_chain,get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
    )

    user_input = st.text_input("Your question:")
    if user_input:
      session_history=get_session_history(session_id)
      response=conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id":session_id}
        },  # constructs a key "abc123" in store.
    )
    st.write(st.session_state.store)
    st.success("Assistant:", response['answer'])
    st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoQ API Key")
     

#There is no temp.pdf file in your folder.

##PyPDFLoader therefore loads zero pages → [].

#then:












                                                                            