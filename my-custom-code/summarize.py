from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
import gradio as gr

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


def embedding_cost(texts: list) -> None:
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total de tokens: {total_tokens}')
    print(f'Custo de Embedding em USD: {total_tokens / 1000 * 0.0001:6f}')


def load_document(file):
    import os
    _, extension = os.path.splitext(file)

    print(f'Carregando arquivo {file}')
    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file) #cada pagina vira um documento LangChain
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    else:
        print('Formato não suportado!')
        return None

    data = loader.load_and_split()
    print(f'Qtidade de documentos: {len(data)}')
    embedding_cost(data)
    return data

#model = 'gpt-3.5-turbo'
model = 'gpt-3.5-turbo-1106'
#model = 'gpt-4-1106-preview'
llm = ChatOpenAI(model_name=model, temperature=0)

def summarize_doc(file_path: str):
    docs = load_document(file_path)
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    summary = chain.invoke(docs)
    return summary['output_text']


input_pdf_path = gr.components.Textbox(label="Provide the PDF file path")
output_summary = gr.components.Textbox(label="Summary")

interface = gr.Interface(
    fn=summarize_doc,
    inputs=input_pdf_path,
    outputs=output_summary,
    title="PDF Summarizer",
    description="Provide PDF file path to get the summary.",
).launch(share=False)