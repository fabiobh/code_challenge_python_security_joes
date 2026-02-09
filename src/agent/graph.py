"""
LangGraph ReAct Agent implementation.

This is the brain of the application! It implements a ReAct (Reasoning + Acting)
agent using LangGraph. The pattern is simple but powerful:

    Think → Act → Observe → Think → Act → Observe → ... → Done

In each cycle:
1. THINK: The LLM looks at the conversation and decides what to do next
2. ACT: If the LLM wants to use a tool, we execute it
3. OBSERVE: The tool result goes back into the conversation
4. Repeat until the LLM has enough info to answer

LangGraph makes this easy by letting us define this as a state machine:
- Nodes: think_node and act_node
- Edges: conditional routing based on what the LLM decided
- State: conversation history, tool calls, current step

The LLM is prompted to always respond with JSON, either:
- {"tool": "name", "params": {...}}  → triggers act node
- {"answer": "..."}                  → we're done!
"""

import os
import json
import logging
from typing import Literal

# nest_asyncio allows nested event loops - needed because LangGraph's
# invoke() is sync but we're already in an async context from main_loop()
import nest_asyncio
nest_asyncio.apply()

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from .state import AgentState, ToolCall
from src.mcp_client.client import MCPClient, MCPToolError, MCPClientError

logger = logging.getLogger(__name__)


# =============================================================================
# System Prompt
# =============================================================================
# This is crucial - it tells the LLM how to behave and what tools are available.
# The LLM doesn't "know" about our tools automatically; we have to explain them.
# I'm very explicit about the expected JSON format because LLMs can be creative
# in ways we don't want here.

SYSTEM_PROMPT = """You are a helpful assistant that can use tools to answer questions about the data center's weather.

The "data center" refers to the machine running the MCP server. To get the weather forecast for the data center, you must call tools in this exact order:

1. **ipify** - First, get the public IP address of the data center
2. **ip_to_geo** - Then, convert that IP to latitude/longitude coordinates  
3. **weather_forecast** - Finally, get the weather forecast for those coordinates

Important rules:
- Always call tools in the correct order (ipify → ip_to_geo → weather_forecast)
- Use the actual values returned from each tool call for the next one
- When you have the final weather data, provide a clear, natural language answer
- Include key weather details: temperature, conditions, humidity, wind speed

To call a tool, respond with a JSON object in this exact format:
```json
{"tool": "tool_name", "params": {"param1": "value1"}}
```

When you have the final answer and don't need any more tools, respond with:
```json
{"answer": "Your final answer here"}
```

Always respond with valid JSON only, no additional text."""


# =============================================================================
# LLM Setup
# =============================================================================

def create_llm():
    """
    Create the LLM instance (Google Gemini).
    
    Using Gemini 2.0 for better response quality and JSON formatting.
    
    Temperature is set to 0 for consistent, deterministic outputs.
    We don't want creativity here, just reliable tool calling.
    """
    # Fetch the API key safely
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    
    from langchain_google_genai import HarmBlockThreshold, HarmCategory

    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0,
        # Disable safety filters to prevent blocking of legitimate tool outputs
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        },
    )

# =============================================================================
# Think Node
# =============================================================================

def think_node(state: AgentState) -> AgentState:
    """
    THINK: The LLM decides what to do next.
    
    This node:
    1. Sends the conversation history to the LLM
    2. Parses the LLM's JSON response
    3. Updates state based on what the LLM wants to do
    
    The LLM will either:
    - Request a tool call → sets pending_tool_call and current_step="act"
    - Provide final answer → sets final_answer and current_step="done"
    
    If the LLM's response isn't valid JSON, we treat it as the final answer.
    This is a fallback to avoid crashing on malformed responses.
    """
    logger.info("Think node: LLM deciding next action")
    
    llm = create_llm()
    
    # Build the full message list with system prompt at the start
    # The system prompt sets up the rules, then comes the conversation
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    
    # Get the LLM's response - try up to 3 times if we get empty response
    response_text = ""
    for attempt in range(3):
        response = llm.invoke(messages)
        response_text = (response.content or "").strip()
        
        if response_text:
            break
        logger.warning(f"LLM returned empty response, attempt {attempt + 1}/3")
    
    logger.info(f"LLM response: {response_text}")
    
    # Handle empty response
    if not response_text:
        logger.error("LLM returned empty response after all retries")
        # Check if we have enough data to provide a weather answer
        tool_calls = state.get("tool_calls", [])
        if len(tool_calls) >= 3:
            # We have weather data, construct an answer
            weather = tool_calls[-1].get("result", {})
            return {
                **state,
                "messages": state["messages"] + [AIMessage(content='{"answer": "Unable to generate response"}')],
                "current_step": "done",
                "final_answer": f"Weather at the data center: {weather.get('temperature_celsius', 'N/A')}°C, {weather.get('weather_description', 'N/A')}. " +
                               f"Wind: {weather.get('wind_speed_kmh', 'N/A')} km/h, Humidity: {weather.get('humidity_percent', 'N/A')}%",
                "pending_tool_call": None,
            }
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content="")],
            "current_step": "done",
            "final_answer": "Sorry, I was unable to get a response from the AI. Please try again.",
            "pending_tool_call": None,
        }
    
    # Parse the JSON response
    try:
        # Clean up the response - sometimes LLMs wrap JSON in markdown code blocks
        clean_text = response_text
        if clean_text.startswith("```"):
            lines = clean_text.split("\n")
            # Strip the opening ``` and closing ``` lines
            if lines[-1].strip() == "```":
                clean_text = "\n".join(lines[1:-1])
            else:
                clean_text = "\n".join(lines[1:])
        
        parsed = json.loads(clean_text)
        
        if "answer" in parsed:
            # LLM is done - it has the final answer
            logger.info("LLM provided final answer")
            return {
                **state,
                "messages": state["messages"] + [AIMessage(content=response_text)],
                "current_step": "done",
                "final_answer": parsed["answer"],
                "pending_tool_call": None,
            }
            
        elif "tool" in parsed:
            # LLM wants to call a tool
            tool_name = parsed["tool"]
            params = parsed.get("params", {})
            logger.info(f"LLM wants to call tool: {tool_name} with params: {params}")
            return {
                **state,
                "messages": state["messages"] + [AIMessage(content=response_text)],
                "current_step": "act",
                "pending_tool_call": {"tool": tool_name, "params": params},
            }
            
        else:
            # Unexpected format - treat as final answer
            logger.warning("Unexpected LLM response format, treating as final answer")
            return {
                **state,
                "messages": state["messages"] + [AIMessage(content=response_text)],
                "current_step": "done",
                "final_answer": response_text,
                "pending_tool_call": None,
            }
            
    except json.JSONDecodeError as e:
        # LLM didn't return valid JSON - use raw text as answer
        # This is a fallback; ideally the prompt prevents this
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=response_text)],
            "current_step": "done",
            "final_answer": response_text,
            "pending_tool_call": None,
        }


# =============================================================================
# Act Node
# =============================================================================

async def act_node(state: AgentState, mcp_client: MCPClient) -> AgentState:
    """
    ACT: Execute the tool that the LLM requested.
    
    This node:
    1. Reads the pending tool call from state
    2. Calls the MCP server to execute the tool
    3. Records the result in the tool_calls trace
    4. Adds the result as a message so the LLM can see it
    5. Goes back to "think" so the LLM can decide what's next
    
    If the tool fails, we still add the error as a message.
    The LLM can then see the error and try something else.
    """
    # Retrieve the tool execution details set by the Think node
    pending = state.get("pending_tool_call")
    if not pending:
        # This shouldn't happen, but handle it gracefully
        logger.error("Act node called but no pending tool call")
        return {**state, "current_step": "think"}
    
    # Unpack the tool name and arguments
    tool_name = pending["tool"]
    params = pending.get("params", {})
    
    logger.info(f"Act node: Executing tool {tool_name}")
    
    try:
        # Call the tool via MCP client
        result = await mcp_client.call_tool(tool_name, params)
        
        # Record successful call in our trace
        tool_call = ToolCall(
            tool_name=tool_name,
            params=params,
            result=result,
            error=None,
        )
        
        # Add the tool result as a message so the LLM can see it
        # We use HumanMessage here because we're using prompt-based tool calling,
        # not native function calling. This treats the tool output as "observation"
        # from the environment.
        from langchain_core.messages import HumanMessage
        result_message = HumanMessage(
            content=f"Tool '{tool_name}' result:\n{json.dumps(result, indent=2)}"
        )
        
        return {
            **state,
            "messages": state["messages"] + [result_message],
            "tool_calls": state["tool_calls"] + [tool_call],
            "current_step": "think",  # Back to thinking
            "pending_tool_call": None,
        }
        
    except (MCPToolError, MCPClientError) as e:
        # Tool call failed - record the error
        logger.error(f"Tool call failed: {e}")
        
        tool_call = ToolCall(
            tool_name=tool_name,
            params=params,
            result=None,
            error=str(e),
        )
        
        # Add error as a message
        from langchain_core.messages import HumanMessage
        error_message = HumanMessage(
            content=f"Tool '{tool_name}' failed with error: {str(e)}"
        )
        
        return {
            **state,
            "messages": state["messages"] + [error_message],
            "tool_calls": state["tool_calls"] + [tool_call],
            "current_step": "think",  # Let LLM handle the error
            "pending_tool_call": None,
        }


# =============================================================================
# Routing Logic
# =============================================================================

def route_after_think(state: AgentState) -> Literal["act", "end"]:
    """
    Decide where to go after the think node.
    
    This is the conditional edge in LangGraph. Based on current_step:
    - "done" → END (we're finished, return the answer)
    - "act" → act node (execute the pending tool)
    - anything else → END (safety fallback)
    """
    if state["current_step"] == "done":
        return "end"
    elif state["current_step"] == "act":
        return "act"
    else:
        return "end"  # Safety fallback


# =============================================================================
# Graph Builder
# =============================================================================

def build_agent_graph(mcp_client: MCPClient) -> StateGraph:
    """
    Build the LangGraph state machine for our ReAct agent.
    
    The graph looks like this:
    
        ┌─────────────┐
        │   START     │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐      ┌─────────────┐
        │   THINK     │─────►│    ACT      │
        │  (LLM)      │◄─────│  (tool)     │
        └──────┬──────┘      └─────────────┘
               │
               ▼ (when done)
        ┌─────────────┐
        │    END      │
        └─────────────┘
    
    The think node decides whether to call a tool or finish.
    The act node executes the tool and goes back to think.
    """
    # Create the graph with our state type
    graph = StateGraph(AgentState)
    
    # Add the nodes
    graph.add_node("think", think_node)
    # We need a sync wrapper for act_node because LangGraph invoke is sync
    graph.add_node("act", lambda state: act_node_sync(state, mcp_client))
    
    # Start at the think node
    graph.set_entry_point("think")
    
    # Add conditional edge from think
    # Based on current_step, go to either "act" or "end"
    graph.add_conditional_edges(
        "think",
        route_after_think,
        {
            "act": "act",
            "end": END,
        }
    )
    
    # After acting, always go back to thinking
    graph.add_edge("act", "think")
    
    # Compile and return the executable graph
    return graph.compile()


def act_node_sync(state: AgentState, mcp_client: MCPClient) -> AgentState:
    """
    Synchronous wrapper for the async act_node.
    
    LangGraph's invoke() is synchronous, but our act_node is async
    (because it makes HTTP calls). This wrapper bridges the gap.
    
    Note: This isn't ideal - mixing sync and async can cause issues.
    In a production app, you might want to use LangGraph's async invoke.
    """
    import asyncio
    
    # Get or create an event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(act_node(state, mcp_client))


# =============================================================================
# Agent Runner
# =============================================================================

async def run_agent(question: str, mcp_client: MCPClient) -> tuple[str, list[ToolCall]]:
    """
    Run the ReAct agent to answer a user's question.
    
    This is the main entry point for using the agent. Give it a question,
    and it will:
    1. Build the LangGraph
    2. Create initial state with the question
    3. Run the graph until it reaches END
    4. Return the final answer and trace of tool calls
    
    Args:
        question: What the user wants to know
        mcp_client: Client for calling MCP tools
    
    Returns:
        Tuple of (answer_text, list_of_tool_calls)
    """
    from .state import create_initial_state
    
    logger.info(f"Running agent with question: {question}")
    
    # Make sure tools are loaded (triggers cache if first time)
    await mcp_client.list_tools()
    
    # Build and run the graph
    graph = build_agent_graph(mcp_client)
    # Initialize the state with the user's input question
    initial_state = create_initial_state(question)
    
    # Run! This executes think→act→think→... until we hit END
    final_state = graph.invoke(initial_state)
    
    return (
        final_state.get("final_answer", "No answer generated"),
        final_state.get("tool_calls", []),
    )
