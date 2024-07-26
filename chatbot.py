
import gradio as gr
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def load_faiss_index(api_key, load_path="faiss_store", allow_dangerous_deserialization=True):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    return FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=allow_dangerous_deserialization)

def query_vectorstore(vectorstore, query):
    results = vectorstore.similarity_search(query)
    return results

def chatbot(api_key, query):
    try:
        vectorstore = load_faiss_index(api_key)
        results = query_vectorstore(vectorstore, query)
        if results:
            return results[0].page_content
        else:
            return "Sorry, I don't have an answer for that."
    except Exception as e:
        return str(e)

iface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Textbox(lines=1, label="OpenAI API Key", type="password"),
        gr.Textbox(lines=1, label="Query")
    ],
    outputs=gr.Textbox(lines=5, label="Response"),
    title="FAISS Chatbot",
    description="Enter your OpenAI API Key and a query to get a response from the chatbot."
)


iface.launch(share=True)
