import asyncio
import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()
client = MultiServerMCPClient(
    {
        "search_mcp": {
            #"url": "http://localhost:8000/mcp",
            #"transport": "streamable_http"
            "command": "python",
            "args": ["mcp_server.py"],
            "transport": "stdio"
        }
    }
)

tools = asyncio.run(client.get_tools())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-5-nano", api_key=OPENAI_API_KEY)

agent = create_agent(llm, tools)

st.title("AI Agent (MCP Version)")
task = st.text_input("Assign me a task")

if task:
    response = asyncio.run(agent.ainvoke({"messages": task}))
    st.write("Agent's response:",response)
    final_output = response["messages"][-1].content
    st.write(final_output)

