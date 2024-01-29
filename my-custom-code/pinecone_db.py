from abstract_vector_db import AbstractVectorDB
from pinecone import Pinecone, PodSpec
from langchain.vectorstores.pinecone import Pinecone as Pineconelg
import os
from langchain_core.vectorstores import VectorStoreRetriever
from pinecone.core.client.exceptions import PineconeApiException

class PineconeDB(AbstractVectorDB):

    def __init__(self) -> None:
        PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
        PINECONE_ENV = os.environ.get('PINECONE_ENV')
        self.pc = Pinecone(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV,
        )
        self.vector_db = None


    def store(self, index_name, chunks, embeddings) -> None:
        try:
            print(f'Criando índice {index_name}')
            self.pc.create_index(
                name=index_name,
                dimension=1536,
                spec=PodSpec(
                    environment="gcp-starter", 
                    pod_type="starter"
                ),
                metric='cosine'
                )
        except PineconeApiException as error:
            print(f'Index already exists: {error.body}')

        try:
            vector_db = Pineconelg.from_documents(
                chunks,
                embeddings,
                index_name=index_name,
            )
            self.vector_db = vector_db
        except PineconeApiException as error:
            print(f'Store index error: {error.body}')
    

    def load(self, index_name, embeddings) -> None:
        try:
            vector_db = Pineconelg.from_existing_index(
                index_name,
                embeddings,
            )
            self.vector_db = vector_db
        except ValueError as error:
            print(f'Load index error: {error}')
            # exit(1)
    

    def list_indexes(self) -> list:
        indexes = self.pc.list_indexes().index_list
        return [index['name'] for index in indexes['indexes']]
    

    def delete_index(self, index_name) -> bool:
        try:
            self.pc.delete_index(index_name)
            return True
        except pinecone.exceptions.NotFoundException:
            return False
    

    def document_exists(self, document_name) -> bool:
        try:
            document_search = self.vector_db._index.query(
                vector=[0] * 1536,
                filter={'source': {'$eq': document_name}},
                top_k=1,
                include_metadata=True
            )['matches']
            return len(document_search) > 0
        except AttributeError as error:
            print(f'No index loaded! {error}')
            # exit(1)
            return False
        except PineconeApiException as error:
            print(f'Error: {error.body}')
    

    def get_retriever(self, search_type: str, search_kwargs: dict) -> VectorStoreRetriever:
        try:
            retriever = self.vector_db.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            return retriever
        except AttributeError as error:
            print(f'Error: Vector DB not loaded: {error}')
            exit(1)