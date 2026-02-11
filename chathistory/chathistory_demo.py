from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory





load_dotenv()
  
llm = ChatOllama(model="llama3.2:3b")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a Agile Coach.Answer any questions "
         "related to agile process"),
         MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}")
    ]
)


chain = prompt_template | llm

history_for_chain = ChatMessageHistory()

chain_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history")
print("Agile Guide")
while True:
    question = input("Enter the question:  ")
    if question:
        response = chain_history.invoke(
            {"input": question},
            config={"configurable": {"session_id": "default"}}
        )
        print(response.content)

