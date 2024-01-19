# NewsGPT
A latest news QA application backed by ChatGPT, LangChain, Pinecone and Kafka

# Run
First install dependencies:
```
pip3 install -r requirements.txt
```
Then run streamlit app
```
streamlit run app.py
```
Then navigate to localhost:8501 to access the app.

# Online Deployment
Now the app has been deployed to AWS! Check it out yourself and stay tuned!
```
http://my-new-lb-2-281783900.us-east-2.elb.amazonaws.com:8501/
```

# News Retrieval
I use a self-written data pipeline for aggregating news from major news provider RSS streams.
[News Aggregator](https://github.com/Alfr3doK1ng/news_aggregator)

# Screenshot
![Alt text](<Screenshot 2024-01-19 at 3.53.59 PM.png>)