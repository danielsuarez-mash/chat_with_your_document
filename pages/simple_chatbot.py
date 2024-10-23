import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
st.title("Simple Chatbot")

# define LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    max_new_tokens=1000,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)

# pipe LLM into chat model class
chat_model = ChatHuggingFace(llm=llm)

# check if message history already exists, if not, create one
if "message_history" not in st.session_state:
    st.session_state.message_history = [
        SystemMessage(
        content="You are a helpful AI assistant. Give your answer in markdown format."
      )
    ]

# Display all human and AI chat messages from history on app 
for message in st.session_state.message_history:
    if isinstance(message, AIMessage):
      with st.chat_message("assistant"):
        st.markdown(message.content)
    elif isinstance(message, HumanMessage):
      with st.chat_message("user"):
        st.markdown(message.content)

# display prompt
if user_input := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # add user input to message history
    st.session_state.message_history.append(HumanMessage(content=user_input))

    # AI response
    response = chat_model.bind(max_tokens=1000, temperature=1).invoke(st.session_state.message_history)
    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.message_history.append(AIMessage(content=response.content))


