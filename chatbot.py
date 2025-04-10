import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true" #to keep the monitoring result
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


from pinecone import ServerlessSpec,Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

st.title("বাঙালি মাস্টার-শেফ")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "ssc-rag-project"
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)


# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

    st.session_state.messages.append(SystemMessage("You are an helpful Bengali masterchef who help people to cook in bengali style food. "))


for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

prompt =  st.chat_input("আমি বাঙালি মাস্টার-শেফ, বলো আজকে কি রান্না করবে?")

if prompt:
    #add the messages user input to message history
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

    #system prompt
    system_prompt= """
    You are an assistant for question-answering as a Bengali masterchef. You will answer the question in Bengali. You will provide the recipe of the dish.
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.
    Context:{context}
    """

    # Populate the system prompt with the retrieved context
    system_prompt_fmt = system_prompt.format(context=docs_text)


    print("-- SYS PROMPT --")
    print(system_prompt_fmt)

    # adding the system prompt to the message history
    st.session_state.messages.append(SystemMessage(system_prompt_fmt))

    # invoking the llm
    result = llm.invoke(st.session_state.messages).content

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(result)

        st.session_state.messages.append(AIMessage(result))