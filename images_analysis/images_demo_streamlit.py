from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import base64
import os
from dotenv import load_dotenv
load_dotenv()

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-5-nano", api_key=OPENAI_API_KEY)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can describe images."),
        (
            "human",
            [
                {"type": "text", "text": "{input}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,""{image}",
                        "detail": "low",
                    },
                },
            ],
        ),
    ]
)

chain = prompt | llm
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
question = st.text_input("Enter your question about the image: ")
if question and uploaded_file:
    image = encode_image(uploaded_file)
    response = chain.invoke({"input": question, "image": image})
    st.write(response.content)