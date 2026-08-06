from langchain_openai import ChatOpenAI
from tools.search import web_search
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model='gpt-5.6-luna', 
    use_responses_api=True,
)

tools = [
    web_search,
]