from langgraph.graph import StateGraph, END, START
from .states import AgentState
from langchain_core.messages import SystemMessage, ToolMessage

class ReActAgent:
    def __init__(self, model, tools, checkpointer=None, system=""):
        self.system = system
        graph = StateGraph(AgentState)

        graph.add_node('llm', self.call_llm)
        graph.add_node('action', self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {
                True: 'action',
                False: END,
            }
        )

        graph.add_edge(START, 'llm')
        graph.add_edge('action', 'llm')

        self.graph = graph.compile(
            checkpointer=checkpointer,
        )

        self.tools = {
            tool.name: tool for tool in tools
        }
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        last_message = state['messages'][-1]
        return len(last_message.tool_calls) > 0

    def call_llm(self, state: AgentState):
        messages = state['messages']

        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        response = self.model.invoke(messages)

        return {
            'messages': [response]
        }

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []

        for tool in tool_calls:
            print(f'Calling: {tool}')

            if not tool['name'] in self.tools:
                print(f'\n ......bad tool name......')
                res = 'bad tool name, retry'
            else:
                res = self.tools[tool['name']].invoke(tool['args'])
            results.append(ToolMessage(tool_call_id=tool['id'], name=tool['name'], content=str(res)))
        print('Back to the model')

        return {
            'messages': results,
        }