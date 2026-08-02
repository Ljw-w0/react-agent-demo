from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from dotenv import load_dotenv
import os

from tools.search import web_search

load_dotenv()

llm_client = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.6-35B-A3B",
    temperature=0.1,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_KEY")
)

tools = [
    web_search,
]

model = ChatHuggingFace(
    llm=llm_client,
).bind_tools(tools)