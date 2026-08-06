from agent.graph import create_agent_graph
from langchain_core.messages import HumanMessage

from models.openai import model, tools
from agent.react_agent import ReActAgent
from prompts.prompts import SYSTEM_PROMPT

import asyncio
import uuid

async def main():

    exits = ['/q', '/quit', '/exit']
    agent = ReActAgent(model=model, tools=tools, system=SYSTEM_PROMPT)
    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4),
        },
    }

    print("\n\n========= Starting agent workflow =========\n\n")

    while True:
        message = input("Enter your query: ")

        if message in exits:
            break

        state = {"messages": [HumanMessage(content=message)]}

        async for event in agent.graph.astream_events(state, thread, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                text = event["data"]["chunk"].text
                if text:
                    # Empty content in the context of OpenAI means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content
                    print(text, end="", flush=True)

        print("\n\n")

if __name__ == "__main__":
    asyncio.run(main())