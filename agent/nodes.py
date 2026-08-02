from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from .states import AgentState
from models.qwen import model, tools
from typing import Any
from prompts.prompts import SYSTEM_PROMPT

tool_node = {
    tool.name: tool for tool in tools
}

def agent_node(state: AgentState) -> dict[str, Any]:
    """
    A simple agent node that processes the current state and returns the next state.

    Args:
        - state (AgentState): The current state of the agent.

    Returns:
        - AgentState: The updated state of the agent after processing.
    """

    last_message = state['messages'][-1]

    if state['current_step'] >= state['max_steps']:
        response = model.invoke(
            [
                SystemMessage(
                    content = f"You have reached the maximum number of steps ({state['max_steps']}). \
                                    Please provide a final answer based on the conversation so far."
                ),
                HumanMessage(content=last_message.content),
            ]
        )

        return {
            'messages': [response],
            'next_action': 'done',
            'final_answer': response.content,
        }
    
    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=last_message.content),
        ]
    )
    next_step = state['current_step'] + 1

    if response.tool_calls:
        return {
            'messages': [response],
            'next_action': 'search',
            'current_step': next_step,
        }

    return {
        'messages': [response],
        'next_action': 'done',
        'current_step': next_step,
        'final_answer': response.content,
    }

def tool_call_node(state: AgentState) -> dict[str, Any]:
    """
    A node that handles tool calls based on the current state.

    Args:
        - state (AgentState): The current state of the agent.

    Returns:
        - AgentState: The updated state of the agent after processing tool calls.
    """

    last_message = state['messages'][-1]

    if not isinstance(last_message, AIMessage):
        raise ValueError("The last message must be an AIMessage to process tool calls.")
    
    if not last_message.tool_calls:
        raise ValueError("No tool calls found in the last AIMessage.")
    
    tool_response = []

    for tool in last_message.tool_calls:
        tool_name = tool['name']
        if tool_name not in tool_node:
            raise ValueError(f"Tool '{tool_name}' is not registered in the tool_node.")
        
        tool_func = tool_node[tool_name]
        result = tool_func.invoke(tool['args'])
        tool_response.append(
            ToolMessage(
                content=result,
                tool_name=tool_name,
                tool_args=tool['args'],
                tool_call_id=tool['id'],
            )
        )

    return {
        'messages': tool_response,
        'next_action': None,
    }


def condition_node(state: AgentState) -> str:
    """
    Condition function: decide next route based on state.

    Args:
        - state (AgentState): The current state of the agent.

    Returns:
        - str: Decision for next route.
    """

    last_message = state["messages"][-1]

    if state['next_action'] != 'done':
        return "continue_to_tool_call"

    return "done"