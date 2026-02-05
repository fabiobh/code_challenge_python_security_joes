"""
LangGraph State definitions for the ReAct agent.

This file defines the data structure that gets passed around in the LangGraph.
Think of state as the "memory" of the agent - it tracks:
- The conversation so far (messages)
- What tools have been called (tool_calls)
- What the agent is currently doing (current_step)
- The final answer once we're done

LangGraph works by having nodes that read and update this state.
Each node gets the current state, does something, and returns the updated state.
"""

from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages

class ToolCall(TypedDict):
    """
    Record of a single tool call made by the agent.
    
    We keep track of every tool call for two reasons:
    1. Debugging - we can see exactly what the agent did
    2. Output - we show the user a trace of tool calls at the end
    """
    tool_name: str       # Which tool was called (e.g., "ipify")
    params: dict         # What parameters were passed
    result: Optional[dict]   # What the tool returned (None if error)
    error: Optional[str]     # Error message if the call failed


class AgentState(TypedDict):
    """
    The main state structure for our ReAct agent.
    
    This is passed through the LangGraph as each node executes.
    It's basically everything the agent needs to know to continue working.
    
    The Annotated[list, add_messages] bit is a LangGraph feature -
    it tells the graph to automatically merge messages instead of
    replacing them. So when a node returns new messages, they get
    appended to the existing list.
    """
    
    # Conversation history with the LLM
    # This includes the user's question, LLM responses, and tool results
    # The add_messages reducer means new messages get appended, not replaced
    messages: Annotated[list, add_messages]
    
    # History of all tool calls made during this query
    # Each entry has the tool name, params, result, and any error
    # We display this at the end so the user can see what happened
    tool_calls: list[ToolCall]
    
    # Where we are in the ReAct loop
    # Values: "think" (LLM deciding), "act" (executing tool), "done" (finished)
    # This is used by the routing logic to decide which node to go to next
    current_step: str
    
    # The final answer to return to the user
    # This gets set when the LLM decides it has enough info to answer
    final_answer: Optional[str]
    
    # Tool call waiting to be executed
    # The think node sets this when the LLM wants to call a tool
    # The act node reads it, executes the tool, and clears it
    pending_tool_call: Optional[dict]


def create_initial_state(user_question: str) -> AgentState:
    """
    Create the starting state for a new query.
    
    This is called at the beginning of each conversation. We set up
    the initial state with the user's question and empty everything else.
    
    Args:
        user_question: What the user asked (e.g., "What's the weather?")
    
    Returns:
        A fresh AgentState ready to start processing
    """
    from langchain_core.messages import HumanMessage
    
    return AgentState(
        messages=[HumanMessage(content=user_question)],  # Start with user's question
        tool_calls=[],           # No tools called yet
        current_step="think",    # Start by thinking about what to do
        final_answer=None,       # No answer yet
        pending_tool_call=None,  # No tool waiting to be called
    )
