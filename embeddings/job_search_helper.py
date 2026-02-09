import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# model = gpt-4o-mini
llm = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

document = TextLoader("../resources/job_listings.txt").load()
text_splitters = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = text_splitters.split_documents(document)

db = Chroma.from_documents(chunks, llm)

query = input("Enter the query:  ")
# Option 1 : Get results by creating embedding vector of query and then invoking similarity search by vector method
# No need to create embedding vector of query, just invoke the retreiver with query
# embedding_vector = llm.embed_query(query)
# results = db.similarity_search_by_vector(embedding_vector)

#Option 2 : Use a retreiver instead of creating embedding of query
# Better approach is to use retreiver instead of creating embedding vector of query and then invoking similarity search by vector method, as retreiver will take care of creating embedding vector of query and then invoking similarity search by vector method internally
retreiver = db.as_retriever()
results = retreiver.invoke(query)

print("Similarity Search Results:", results)
print("Top 3 similar chunks:")
for i, result in enumerate(results[:4]):
    print(f"Chunk {i+1}: {result.page_content}")