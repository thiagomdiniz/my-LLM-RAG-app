import os
from pinecone import Pinecone, PodSpec
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone as Pineconelg
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')

embeddings = OpenAIEmbeddings()

loader = PyPDFLoader("docs/CLT.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

pinecone = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index_name = 'linuxtips'
indexes = pinecone.list_indexes().index_list
if index_name not in [index['name'] for index in indexes['indexes']]:
    pinecone.create_index(
        index_name,
        dimension=1536,
        spec=PodSpec(
            environment="gcp-starter", 
            pod_type="starter"
        )
    )

index = pinecone.Index(index_name)

docsearch = Pineconelg.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

# Test
# query = 'o que são as férias?'
# docs = docsearch.similarity_search(query)
# print(docs[0])