from abstract_vector_db import AbstractVectorDB
import os
from langchain.vectorstores.chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from chromadb.errors import InvalidDimensionException

class ChromaDB(AbstractVectorDB):

    def __init__(self) -> None:
        CHROMA_PATH = os.environ.get('CHROMA_PATH')
        self.vectordb_path = CHROMA_PATH
        self.vector_db = None
    

    def store(self, index_name, chunks, embeddings) -> None:
        try:
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.vectordb_path,
                collection_name=index_name
            )
            vector_db.persist()
            self.vector_db = vector_db
        except InvalidDimensionException as error:
            print(f'Error: {error}')
            exit(1)
    

    def load(self, index_name, embeddings) -> None:
        vector_db = Chroma(
            persist_directory=self.vectordb_path,
            embedding_function=embeddings,
            collection_name=index_name,
        )
        self.vector_db = vector_db
    

    def list_indexes(self) -> list:
        collections = []

        for collection in Chroma(
                            persist_directory=self.vectordb_path
                        )._client.list_collections():
            collection_name = collection.model_dump()['name']
            collections.append(collection_name)

        return collections
    

    def delete_index(self, index_name) -> bool:
        try:
            Chroma(
                persist_directory=self.vectordb_path
            )._client.delete_collection(index_name)
            return True
        except ValueError:
            return False
    

    def document_exists(self, document_name) -> bool:
        try:
            document_search = self.vector_db.get(
                where={'source': document_name},
                limit=1
            )['ids']
            return len(document_search) > 0
        except AttributeError as error:
            print(f'No index loaded! {error}')
            # exit(1)
            return False
    

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