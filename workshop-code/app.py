import os
from pinecone import Pinecone
import gradio as gr
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.vectorstores.pinecone import Pinecone as Pineconelg
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')
pinecone = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

embeddings = OpenAIEmbeddings()

index_name = 'linuxtips'
index = pinecone.Index(index_name)

llm = ChatOpenAI(model='gpt-4-1106-preview', temperature=0)

template="""Assistente é uma IA jurídica que tira dúvidas.
    Assistente elabora respostas simplificadas, com base no contexto fornecido.
    Assistente fornece referências extraídas do contexto abaixo. Não gere links ou referências adicionais.
    Ao final da resposta exiba no formato de lista as referências extraídas.
    Caso não consiga encontrar no contexto abaixo ou caso a pergunta não esteja relacionada do contexto jurídico,
    diga apenas 'Eu não sei!'.

    Pergunta: {query}

    Contexto: {context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=['query', 'context']
)

def search(query: str):
    docsearch = Pineconelg.from_existing_index(embedding=embeddings, index_name=index_name)
    docs = docsearch.similarity_search(query, k=3)
    context = docs[0].page_content + docs[1].page_content + docs[2].page_content
    resp = LLMChain(prompt=prompt, llm=llm)
    return resp.invoke({'query': query, 'context': context})['text']

with gr.Blocks(title='IA jurídica', theme=gr.themes.Soft()) as ui:
    gr.Markdown('# Sou uma IA que tem a CLT como base de conhecimento')
    query = gr.Textbox(label='Faça a sua pergunta:', placeholder='Ex: como funcionam as férias do trabalhador?')
    text_output = gr.Textbox(label='Resposta')
    btn = gr.Button('Perguntar')
    btn.click(fn=search, inputs=query, outputs=[text_output])
ui.launch(debug=True, share=False)