from typing import TypedDict, Literal, Any, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    current_step: int
    max_steps: int
    next_action: Literal["search", "done"] | None
    user_query: str
    search_results: str
    final_answer: str