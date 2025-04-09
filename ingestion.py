import os
from dotenv import load_dotenv
load_dotenv()
import time

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true" #to keep the monitoring result
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

from pinecone import Pinecone,ServerlessSpec

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "ssc-rag-project"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = PineconeVectorStore(index=index, embedding=embedding)

loader = PyPDFDirectoryLoader("Books/")
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20,
)

documents = text_splitter.split_documents(raw_documents)
# print(documents)

# generating id for each entry

i=0
id=[]

while i<len(documents):
    id.append(str(i))
    i+=1

# vector_store.add_documents(documents=documents, ids=id)

from tqdm import tqdm  # Optional: for showing progress bar

batch_size = 100  # You can tune this number
total_docs = len(documents)

for i in tqdm(range(0, total_docs, batch_size)):
    batch_docs = documents[i:i+batch_size]
    batch_ids = id[i:i+batch_size]
    vector_store.add_documents(documents=batch_docs, ids=batch_ids)