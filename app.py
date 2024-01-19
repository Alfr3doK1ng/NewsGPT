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
import time

# Function to get the latest news
def get_latest_news():
    try:
        response = requests.get("https://f881-75-6-176-129.ngrok-free.app/latest_news")  # Update news with Djagon app URL
        if response.status_code == 200:
            return response.json()[:5]  # Get top 5 news items
        else:
            st.error("Failed to retrieve news: HTTP Status Code {}".format(response.status_code))
    except requests.exceptions.RequestException as e:
        st.error(f"Error during requests to localhost:8000/latest_news: {str(e)}")
        return None
    
hover_css = """
<style>
.hover-lift img {
    transition: transform 0.3s ease-in-out;
}

.hover-lift img:hover {
    transform: translateY(-10px);
}
</style>
"""
st.markdown(hover_css, unsafe_allow_html=True)


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
        with st.spinner('Thinking...'):
            answer = chain.run(full_query)

        st.divider()
        
        value = f'Q: {query}\n A: {answer}'
        st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
        st.text_area(label='Chat History', value=st.session_state.history, key='history', height=400)
    

    # Display latest news in sidebar
    latest_news = get_latest_news()


    # Sidebar with news items
    st.sidebar.markdown(f"# Latest News")
    st.sidebar.write(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")
    st.sidebar.divider()

    if st.sidebar.button('Refresh News'):
        latest_news = get_latest_news()

    for news_item in latest_news:
        # Display the title
        st.sidebar.markdown(f"{news_item['title']}")

        # Custom markdown to make the image a clickable link
        st.sidebar.markdown(
            f"<a href='{news_item['news_url']}' target='_blank'>" + 
            f"<div class='hover-lift'><img src='{news_item['image_url']}' width='100%'></div>",
            unsafe_allow_html=True
        )

        # Use an expander for the content
        with st.sidebar.expander("Show More"):
            st.write(news_item["content"])
    
