import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tracers import ConsoleCallbackHandler

st.title('Advanced Chatbot - chat with your PDF')

# PDF upload
with st.sidebar:

    # user-input
    st.session_state.pdf = st.file_uploader(label='Upload PDF')

def check_pdf_exists():

    # save file to temporary location
    if st.session_state.pdf:
        temp_dir = tempfile.mkdtemp()
        st.session_state.pdf_path = os.path.join(temp_dir, st.session_state.pdf.name)
        with open(st.session_state.pdf_path, "wb") as f:
                f.write(st.session_state.pdf.getvalue())
    
        return True
    
    else: return False

def build_retriever():

    # load
    loader = PyPDFLoader(st.session_state.pdf_path)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)

    # split 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(pages)

    # store
    load_dotenv()
    hf_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    embeddings_model = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    vector_store = FAISS.from_documents(documents=all_splits, embedding=embeddings_model)

    # retriever
    retriever = vector_store.as_retriever()

    return retriever

def instantiate_llm():

    # LLM
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        max_new_tokens=1000,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.01,
        repetition_penalty=1.03,
    )
    chat_model = ChatHuggingFace(llm=llm)
    
    return chat_model

def build_history_aware_retriever(chat_model, retriever):

    # history aware retriever
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = contextualize_q_prompt | chat_model.bind(max_tokens=1000, temperature=1) | StrOutputParser() | retriever

    return history_aware_retriever

def build_qa_chain(chat_model):

    # question answering chain
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
        "You are also given chat history below to help contextualise the question."
        "{chat_history}"
        "{input}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", system_prompt)
        ]
    )
    qa_chain = qa_prompt | chat_model.bind(max_tokens=1000, temperature=1) | StrOutputParser()

    return qa_chain

def build_retrieval_chain(history_aware_retriever, qa_chain):

    # retrieval chain
    retrieval_chain = (
        {"context": history_aware_retriever, "chat_history": RunnablePassthrough(), "input": RunnablePassthrough()}
        | qa_chain
    )

    return retrieval_chain

def instantiate_chat_history():

    # initialise chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def main():

    if check_pdf_exists():

        # get retriever
        st.session_state.retriever = build_retriever()

        # get chat model
        st.session_state.chat_model = instantiate_llm()

        # get history aware retriever
        st.session_state.history_aware_retriever = build_history_aware_retriever(st.session_state.chat_model, st.session_state.retriever)

        # get qa chain
        st.session_state.qa_chain = build_qa_chain(st.session_state.chat_model)

        # get retrieval chain
        st.session_state.retrieval_chain = build_retrieval_chain(st.session_state.history_aware_retriever, st.session_state.qa_chain)

        # instantiate chat history if necessary
        instantiate_chat_history()

        # Display chat messages from history on app rerun
        for message in st.session_state.chat_history:
            with st.chat_message(message.type):
                st.markdown(message.content)

        # putting it all together
        if input := st.chat_input("Say something", key=2):
            with st.chat_message("human"):
                st.markdown(input)

            # AI response
            response = st.session_state.retrieval_chain.invoke(
                    {"chat_history": st.session_state.chat_history, "input": input}
                )
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # add input and response to message history
            st.session_state.chat_history.extend([HumanMessage(content=input)])
            st.session_state.chat_history.extend([AIMessage(content=response)])

if __name__ == "__main__":
    main()