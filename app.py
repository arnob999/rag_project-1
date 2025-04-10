import streamlit as st
import os


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

from pinecone import ServerlessSpec, Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

st.title("বাঙালি মাস্টার-শেফ")


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "ssc-rag-project"
index = pc.Index(index_name)


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

#chat history keeping only in session
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SystemMessage("You are an helpful Bengali masterchef who help people to cook in bengali style food."))


for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


prompt = st.chat_input("আমি বাঙালি মাস্টার-শেফ, বলো আজকে কি রান্না করবে?")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt))


    llm = ChatOpenAI(model="gpt-4o")


    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    docs = retriever.invoke(prompt)
    docs_text = "".join(doc.page_content for doc in docs)


    system_prompt = f"""
    You are an assistant for question-answering as a Bengali masterchef. You will answer the question in Bengali. You will provide the recipe of the dish.
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.
    Context: {docs_text}
    """

    print("-- SYS PROMPT --")
    print(system_prompt)

    st.session_state.messages.append(SystemMessage(system_prompt))

    result = llm.invoke(st.session_state.messages).content

    with st.chat_message("assistant"):
        st.markdown(result)
    st.session_state.messages.append(AIMessage(result))
