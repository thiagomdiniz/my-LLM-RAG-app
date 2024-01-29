from abc import ABC, abstractmethod
from langchain_core.vectorstores import VectorStoreRetriever

class AbstractVectorDB(ABC):

    @abstractmethod
    def delete_index(self, index_name) -> bool:
        pass

    @abstractmethod
    def list_indexes(self) -> list:
        pass

    @abstractmethod
    def store(self, index_name, chunks, embeddings) -> None:
        pass

    @abstractmethod
    def load(self, index_name, embeddings) -> None:
        pass

    @abstractmethod
    def get_retriever(self, search_type: str, search_kwargs: dict) -> VectorStoreRetriever:
        pass

    @abstractmethod
    def document_exists(self, document_name) -> bool:
        pass