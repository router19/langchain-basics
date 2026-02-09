from langchain_ollama import OllamaEmbeddings
import numpy as np

llm = OllamaEmbeddings(model="llama3.2")

text1 = input("Enter the text1:  ")
text2 = input("Enter the text2:  ")
response1 = llm.embed_query(text1)

response2 = llm.embed_query(text2)
print("Similarity Score: ",np.dot(response1, response2) * 100 ,"%")