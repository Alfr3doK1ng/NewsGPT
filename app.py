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

    llm = ChatOpenAI(model='gpt-4', temperature = 1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 15})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever = retriever)
    if 'history' not in st.session_state:
                st.session_state.history = ''

    query = st.text_input('Ask a question about the content of recent news.')
    if query:

        full_query = f"{st.session_state.history}\nUser: {query}"

        # result = vector_store.similarity_search(query)
        answer = chain.run(full_query)
        st.text_area('LLM Answer: ', value=answer, height = 200)

        st.divider()
        
        value = f'Q: {query}\n A: {answer}'
        st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
        st.text_area(label='Chat History', value=st.session_state.history, key='history', height=400)