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
        ("system", "You are a helpful assistant that can verify identification documents."),
        (
            "human",
            [
                {"type": "text", "text": "Verify the identification details"},
                {"type": "text", "text": "Name: {user_name}"},
                {"type": "text", "text": "DOB:{user_dob}"},
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

st.title("KYC Document Validator")
st.write("Upload your identification document")
chain = prompt | llm
uploaded_file = st.file_uploader("Upload your document", type=["jpg", "jpeg", "png"])

user_name = st.text_input("Enter your name: ")
user_dob = st.text_input("Enter your date of birth (YYYY-MM-DD): ")
if user_name and user_dob and uploaded_file is not None:
    image = encode_image(uploaded_file)
    response = chain.invoke({"user_name": user_name, "user_dob": user_dob, "image": image})
    st.write(response.content)