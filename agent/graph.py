from langgraph.graph import StateGraph, END, START
from .states import AgentState
from .nodes import agent_node, tool_call_node, condition_node
from langchain_core.messages import HumanMessage

def create_agent_graph():

    workflow = StateGraph(AgentState)

    workflow.add_node("tool_call", tool_call_node)
    workflow.add_node("agent", agent_node)

    workflow.add_edge(START, "agent")
    workflow.add_edge("tool_call", "agent")

    workflow.add_conditional_edges(
        "agent",
        condition_node,
        {
            "continue_to_tool_call": "tool_call",
            "done": END
        }
    )

    app = workflow.compile()

    input = {
        'messages': [
            HumanMessage(content="What is the current price of Tesla stock?")
        ],
        'current_step': 0,
        'max_steps': 5,
        'next_action': None,
        'final_answer': None,
    }

    print("\n\n========= Starting agent workflow =========\n\n")

    output = app.invoke(input)

    return output['final_answer']