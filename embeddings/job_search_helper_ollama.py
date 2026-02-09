import os

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


llm = OllamaEmbeddings(model="llama3.2")

document = TextLoader("../resources/job_listings.txt").load()
text_splitters = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = text_splitters.split_documents(document)

db = Chroma.from_documents(chunks, llm)

query = input("Enter the query:  ")
embedding_vector = llm.embed_query(query)
results = db.similarity_search_by_vector(embedding_vector)

print("Similarity Search Results:", results)
print("Top 3 similar chunks:")
for i, result in enumerate(results[:4]):
    print(f"Chunk {i+1}: {result.page_content}")