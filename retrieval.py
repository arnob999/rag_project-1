import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true" #to keep the monitoring result
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

from pinecone import Pinecone,ServerlessSpec

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

#initializing
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

#choosing the index
index_name = "ssc-rag-project"
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

#retrieval
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
    )

result =  retriever.invoke("মুরগির রেসিপি")

for res in result:
    print({res.page_content})