from dotenv import load_dotenv
import os
from typing import List
from abstract_vector_db import AbstractVectorDB
from langchain_core.documents import Document
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from chromadb.errors import InvalidDimensionException
from pinecone.core.client.exceptions import PineconeApiException

class App:

    def __init__(self) -> None:
        self.template="""Answer the user's question based on the context provided.
            At the end of each answer, display the extracted references in list format.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Question: {question}

            Context: {context}
        """
        self.db = self.get_vector_db()
        self.instantiate_llm()


    def get_vector_db(self) -> AbstractVectorDB:
        """
        Return the right vector database based on .env
        """
        if os.environ.get('PINECONE_API_KEY'):
            print('Using Pinecone vector db')
            from pinecone_db import PineconeDB
            return PineconeDB()
        else:
            print('Using Chroma vector db')
            from chroma_db import ChromaDB
            return ChromaDB()
    

    def instantiate_retriever(self,
                              search_type: str = 'similarity',
                              search_kwargs: dict = {'k': 3}
                            ) -> None:
        self.retriever = self.db.get_retriever(search_type, search_kwargs)
    

    def instantiate_llm(self):
        llm_temperature = os.environ.get('LLM_TEMPERATURE')
        from langchain.prompts import PromptTemplate
        self.prompt_template = PromptTemplate.from_template(self.template)

        if os.environ.get('OPENAI_API_KEY'):
            print('Using OpenAI')
            embedding_model = os.environ.get('OPENAI_EMBEDDING_MODEL')
            model = os.environ.get('OPENAI_MODEL_NAME')
            from langchain_openai import OpenAIEmbeddings
            from langchain_openai import ChatOpenAI
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
            self.llm = ChatOpenAI(model_name=model, temperature=llm_temperature)
        else:
            print('Using local Llama')
            model_path = os.environ.get('LLAMA_MODEL_PATH')
            embedding_model = os.environ.get('LOCAL_EMBEDDING_MODEL')
            model_context_length = os.environ.get('LLAMA_MODEL_CONTEXT_LENGTH')
            from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
            from langchain_community.llms import LlamaCpp
            self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
            # from langchain_community.embeddings import LlamaCppEmbeddings
            # self.embeddings = LlamaCppEmbeddings(model_path=model_path)
            self.llm = LlamaCpp(model_path=model_path, verbose=True, n_ctx=model_context_length, temperature=llm_temperature)


    def db_list_indexes(self) -> list:
        return self.db.list_indexes()
    

    def db_delete_index(self, index_name) -> bool:
        return self.db.delete_index(index_name)
    

    def db_store_in_index(self, index_name, chunks) -> None:
        document_name = chunks[0].metadata['source']
        if not self.db.document_exists(document_name):
            self.db.store(index_name, chunks, self.embeddings)
            print('Document stored in vector db!')
        else:
            print('Document already exists in vector db!')
    

    def db_load_from_index(self, index_name) -> None:
        self.db.load(index_name, self.embeddings)
    

    def load_document(self, file: str) -> List[Document] | None:
        _, extension = os.path.splitext(file)

        print(f'Carregando arquivo {file}')
        if extension == '.pdf':
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file)
        elif extension == '.docx':
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file)
        elif extension == '.txt':
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file)
        elif extension == '.csv':
            from langchain_community.document_loaders.csv_loader import CSVLoader
            loader = CSVLoader(file)
        else:
            print('Formato não suportado!')
            return None

        data = loader.load()
        print(f'Qtidade de documentos: {len(data)}')
        return data
    

    def load_documents(self, dir_path: str, index_name: str) -> None:
        for filename in os.listdir(dir_path):
            file = os.path.join(dir_path, filename)
            if os.path.isfile(file):
                document = self.load_document(file)
                chunks = self.chunk_data(document)
                self.embedding_cost(chunks)
                self.db_store_in_index(index_name, chunks)
    

    def chunk_data(self, data, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
        chunks = text_splitter.split_documents(data)
        print(f'Vector/chunk count: {len(chunks)}')
        return chunks


    def embedding_cost(self, texts: list) -> None:
        import tiktoken
        enc = tiktoken.encoding_for_model('text-embedding-ada-002')
        total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
        print(f'Total de tokens: {total_tokens}')
        print(f'Custo de Embedding em USD: {total_tokens / 1000 * 0.0001:6f}')
    

    def __format_docs(self, docs) -> str:
            ctx = "\n\n".join(doc.page_content for doc in docs)
            print(f'\n\nContext: {ctx}')
            return ctx
    

    def search(self, question: str, history):
        try:
            # docs = self.db.vector_db.similarity_search(question, k=3)        
            rag_chain = (
                {"context": self.retriever | self.__format_docs, "question": RunnablePassthrough()}
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )

            ans = ""
            print(f'\n\nQuestion: {question}')
            for chunk in rag_chain.stream(question):
                ans += chunk
                yield ans
        
        except AttributeError as error:
            yield f'Error: Vector DB not loaded: {error}'
        except InvalidDimensionException as error:
            yield f'Error: {error}'
        except PineconeApiException as error:
            yield f'Error: {error.body}'
    

    def start_chat_web_interface(
            self,
            share: bool = False,
            server: str = '127.0.0.1',
            port: int = 7860
        ) -> None:
        self.instantiate_retriever()
        import gradio as gr
        gr.ChatInterface(
            self.search,
            title='My LLM Chat',
            description='https://github.com/thiagomdiniz/my-LLM-playground'
        ).queue().launch(
                share=share,
                server_name=server,
                server_port=port
            )


if __name__ == '__main__':
    load_dotenv(override=True)
        
    app = App()

    print(f'Indexes list: {app.db_list_indexes()}')
    # app.db_delete_index('myindex')
    app.db_load_from_index('myindex')

    # try to load all documents inside a folder:
    app.load_documents(os.environ.get('DOCUMENTS_FOLDER'), 'myindex')

    # load a specific document:
    # document = app.load_document('docs/Boily Public Health.pdf')
    # chunks = app.chunk_data(document)
    # app.embedding_cost(chunks)
    # app.db_store_in_index('myindex', chunks)

    app.start_chat_web_interface()
