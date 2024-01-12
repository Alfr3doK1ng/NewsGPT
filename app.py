import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override = True)

import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

import streamlit as st

if __name__ == "__main__":
    
    embeddings = OpenAIEmbeddings()

    index_name = "news"
    pinecone.init(api_key = os.environ.get('PINECONE_API_KEY'), environment = os.environ.get('PINECONE_ENV'))
    vector_store = Pinecone.from_existing_index(index_name, embeddings)
    st.session_state.vs = vector_store

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature = 1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 10})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever = retriever)

    query = st.text_input('Ask a question about the content of recent news.')
    if query:
        result = vector_store.similarity_search(query)
        answer = chain.run(query)
        st.text_area('LLM Answer: ', value=answer)
