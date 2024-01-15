import os
from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv(), override = True)

import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

import streamlit as st

import requests

# Function to get the latest news
def get_latest_news():
    try:
        response = requests.get("http://localhost:8000/latest_news/")  # Update with your correct URL
        if response.status_code == 200:
            return response.json()[:3]  # Get top 3 news items
        else:
            st.error("Failed to retrieve news: HTTP Status Code {}".format(response.status_code))
    except requests.exceptions.RequestException as e:
        st.error(f"Error during requests to {url}: {str(e)}")
        return None

if __name__ == "__main__":

    st.title('NewsGPT')
    st.subheader('A 🦜🔗 application backed by Django, Kafka and Streamlit')
    
    embeddings = OpenAIEmbeddings(openai_api_key = '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')

    index_name = "news"
    # pinecone.init(api_key = os.environ.get('PINECONE_API_KEY'), environment = os.environ.get('PINECONE_ENV'))
    pinecone.init(api_key = '7a27aee2-c409-4a5c-9fa5-4870382fbf7f', environment = 'gcp-starter')
    vector_store = Pinecone.from_existing_index(index_name, embeddings)
    st.session_state.vs = vector_store

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature = 0.2, openai_api_key = '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')
    # llm = ChatOpenAI(model='gpt-4', temperature = 1, openai_api_key = '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')
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 15})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever = retriever)

    query = st.text_input('Start a conversation with recent news.')

    if 'history' not in st.session_state:
        st.session_state.history = ''
  
    if query: 

        
        full_query = f"{st.session_state.history} User: {query}"
        # st.text_area('full_query', value = full_query)
        with st.spinner('Thinking...'):
        # result = vector_store.similarity_search(query)
            answer = chain.run(full_query)
        # st.text_area('LLM Answer: ', value=answer, height = 200)

        st.divider()
        
        value = f'Q: {query}\n A: {answer}'
        st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
        st.text_area(label='Chat History', value=st.session_state.history, key='history', height=400)
    
    # # Sidebar - Latest News
    # st.sidebar.title("Latest News")
    # if st.sidebar.button('Load Latest News'):
    #     latest_news = get_latest_news()
    #     if latest_news:
    #         for news_item in latest_news:
    #             st.sidebar.markdown(f"**{news_item['title']}**")
    #             st.sidebar.info(news_item['content'])
    #             st.sidebar.write("---")



# import os
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv(), override = True)

# import pinecone
# from langchain.vectorstores import Pinecone
# from langchain.embeddings import OpenAIEmbeddings

# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI

# import streamlit as st

# if __name__ == "__main__":
    
#     embeddings = OpenAIEmbeddings()

#     index_name = "news"
#     pinecone.init(api_key = os.environ.get('PINECONE_API_KEY'), environment = os.environ.get('PINECONE_ENV'))
#     vector_store = Pinecone.from_existing_index(index_name, embeddings)
#     st.session_state.vs = vector_store

#     llm = ChatOpenAI(model='gpt-3.5-turbo', temperature = 1)
#     retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 10})
#     chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever = retriever)

#     query = st.text_input('Ask a question about the content of recent news.')
#     if query:
#         result = vector_store.similarity_search(query)
#         answer = chain.run(query)
#         st.text_area('LLM Answer: ', value=answer)