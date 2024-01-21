import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.title('LLM - Retrieval Augmented Generation')

# user-input
pdf = st.file_uploader(label='Upload PDF')

# sidebar parameters
with st.sidebar:
    chunk_size = st.number_input(label='Chunk size', value=500, step=10)
    chunk_overlap = st.number_input(label='Chunk overlap', value=20, step=10)

# question
question = st.text_input(label='Question')

def authenticate():

    try:
        st.write('Authenticated with HuggingFace:', 
                 os.environ["HUGGINGFACEHUB_API_TOKEN"] == st.secrets["HUGGINGFACEHUB_API_TOKEN"])
    except:
        st.write('Cannot find HugginFace API token. Ensure it is located in .streamlit/secrets.toml')

def load_pdf(pdf):
    
    reader = PdfReader(pdf)

    # page_limit = st.number_input(label='Page limit', value=len(reader.pages), step=1)
    page_limit = len(reader.pages)

    if page_limit is None:
        page_limit=len(reader.pages)
    
    text = ""

    for i in range(page_limit):

        page_text = reader.pages[i].extract_text()

        text += page_text
    
    # if st.toggle(label='Show text'):
    #     st.write(text)
    
    return text

def split_text(text, chunk_size=400, chunk_overlap=20):

    # split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    # use text_splitter to split text
    chunks = text_splitter.split_text(text)

    return chunks

def store_text(chunks):

    # select model to create embeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-large')

    # select vectorstore, define text chunks and embeddings model
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vectorstore

@st.cache_resource
def load_split_store(pdf, chunk_size, chunk_overlap):
    
    # load split store
    text = load_pdf(pdf=pdf)
    chunks = split_text(text, chunk_size, chunk_overlap)
    vectorstore = store_text(chunks)

    return vectorstore

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

def main():

    # authenticate
    authenticate()

    # define new template for RAG
    rag_template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """

    # instantiate llm
    llm = HuggingFaceHub(
        repo_id='tiiuae/falcon-7b-instruct',
        model_kwargs={
            # 'temperature':1,
            # 'penalty_alpha':2,
            # 'top_k':50,
            # # 'max_length': 1000
        }
    )

    # build prompt
    prompt = PromptTemplate(
        template=rag_template, 
        llm=llm, 
        input_variables=['question', 'context']
    )
    
    # if a PDF exists
    if pdf is not None:

        # load split store
        vectorstore = load_split_store(pdf, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        st.write('PDF processed')

        # create a retriever using vectorstore
        retriever = vectorstore.as_retriever()
        
        # create retrieval chain
        retrieval_chain = (
            retriever | format_docs
        )
        
        # create generation chain
        generation_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # button press
        if st.button(label='Ask question'):
            with st.spinner('Processing'):

                # context
                st.write('# Context')
                st.write(retrieval_chain.invoke(question))

                # answer
                st.write('# Answer')
                st.write(generation_chain.invoke(question))



if __name__=='__main__':
    main()


