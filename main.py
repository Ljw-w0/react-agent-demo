from agent.graph import create_agent_graph
from models.qwen import model
from langchain_core.messages import HumanMessage, AIMessage



def main():
    final_answer = create_agent_graph()
    print(f"Final Answer: {final_answer}")

if __name__ == "__main__":
    main()