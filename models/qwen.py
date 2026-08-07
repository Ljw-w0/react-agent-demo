from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from dotenv import load_dotenv
import os

load_dotenv()

llm_client = HuggingFaceEndpoint(
    repo_id=os.getenv("QWEN_MODEL_ID"),
    temperature=0.1,
    max_new_tokens=2048,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_KEY"),
)

model = ChatHuggingFace(
    llm=llm_client,
)