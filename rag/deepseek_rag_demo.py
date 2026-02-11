from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import ChatPromptTemplate


llm = ChatOllama(model="deepseek-r1:7b")
embedding = OllamaEmbeddings(model="deepseek-r1:7b")

document = TextLoader("../resources/product-data.txt").load()
text_splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitters.split_documents(document)

vector_store = Chroma.from_documents(chunks, embedding)
retreiver = vector_store.as_retriever()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","""
            You are an assistant for answering questions.
            Use the provided context to respond.If the answer
            isn't clear, acknowledge that you don't know.
            Limit your response to three concise sentences.
            {context} """),
        
        ("human","{input}")
    ]
    )

qa_chain = create_stuff_documents_chain(llm,prompt_template)
rag_chain = create_retrieval_chain(retreiver,qa_chain)

print("Chat with Document: ")
while True:
    question = input("Enter your question: ")

    if question.lower() == "exit":
        break

    if question:
        response = rag_chain.invoke({"input": question})
        print(response["answer"])