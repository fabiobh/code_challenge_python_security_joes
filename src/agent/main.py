"""
CLI Entry Point for the LangGraph ReAct Agent.

This is where users interact with our agent. It's a simple command-line
interface that:
1. Connects to the MCP server
2. Accepts questions from the user
3. Runs the agent to get answers
4. Displays the result with a trace of tool calls

The terminal-based approach keeps things simple - no web UI to set up,
just run the script and start asking questions!

Usage:
    python -m src.agent.main
    
Then type your questions and see the agent work its magic.
"""

import asyncio
import os
import sys
import logging
from dotenv import load_dotenv

# Add src to path so we can import our modules
# (needed when running as python -m src.agent.main)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.mcp_client.client import MCPClient
from src.agent.graph import run_agent
from src.agent.state import ToolCall


# =============================================================================
# CLI Formatting Functions
# =============================================================================
# These make the output look nice in the terminal.
# Nothing fancy, just clear and readable.

def print_header():
    """Print a welcome message when the CLI starts."""
    print("\n" + "=" * 60)
    print("   MCP Weather Agent - LangGraph ReAct")
    print("=" * 60)
    print("\nAsk me about the weather at the data center!")
    print("Type 'quit' or 'exit' to end the session.\n")


def print_tool_trace(tool_calls: list[ToolCall]):
    """
    Print the trace of tool calls made during the query.
    
    This shows the user exactly what the agent did to get the answer.
    Each step shows the tool name, parameters, and result.
    It's great for debugging and for understanding how the agent works.
    """
    if not tool_calls:
        print("\n📋 Tool Call Trace: (no tools called)")
        return
    
    print("\n📋 Tool Call Trace:")
    print("-" * 40)
    
    for i, call in enumerate(tool_calls, 1):
        # Show checkmark for success, X for failure
        status = "✅" if call.get("error") is None else "❌"
        print(f"\n{status} Step {i}: {call['tool_name']}")
        
        # Show parameters if any were passed
        if call.get("params"):
            print(f"   Params: {call['params']}")
        
        # Show the result (formatted nicely for dicts)
        if call.get("result"):
            result = call["result"]
            if isinstance(result, dict):
                for key, value in result.items():
                    print(f"   {key}: {value}")
            else:
                print(f"   Result: {result}")
        
        # Show error if the tool failed
        if call.get("error"):
            print(f"   Error: {call['error']}")
    
    print("-" * 40)


def print_answer(answer: str):
    """Print the final answer from the agent."""
    print("\n🌤️  Answer:")
    print("-" * 40)
    print(answer)
    print("-" * 40 + "\n")


# =============================================================================
# Main Loop
# =============================================================================
# The heart of the CLI - connects to server, takes questions, runs agent.

async def check_server_health(mcp_client: MCPClient) -> bool:
    """
    Verify the MCP server is running before we start.
    
    This saves the user from typing a question only to get a connection error.
    Better to catch it early and give a helpful message.
    """
    print("Checking MCP server connection...", end=" ")
    if await mcp_client.health_check():
        print("✅ Connected")
        return True
    else:
        print("❌ Failed")
        print("\n⚠️  Could not connect to MCP server.")
        print("   Please start the server first with:")
        print("   python -m src.mcp_server.main")
        return False


async def process_question(question: str, mcp_client: MCPClient):
    """
    Process a single question from the user.
    
    This calls the agent, waits for the result, and displays it.
    Any errors are caught and displayed nicely.
    """
    print("\n🔄 Processing...")
    
    try:
        # Run the agent - this is where the magic happens!
        answer, tool_calls = await run_agent(question, mcp_client)
        
        # Show the results to the user
        print_tool_trace(tool_calls)
        print_answer(answer)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.exception("Agent error")


async def main_loop():
    """
    Main interactive loop - the core of the CLI.
    
    This:
    1. Loads environment variables (for API keys)
    2. Creates the MCP client
    3. Checks the server is running
    4. Loops forever taking questions until user quits
    """
    # Load .env file if present (for GOOGLE_API_KEY)
    load_dotenv()
    
    # Check we have the API key before going further
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n❌ Error: GOOGLE_API_KEY environment variable is not set.")
        print("   Please set it in a .env file or as an environment variable.")
        print("   Get your API key at: https://aistudio.google.com/app/apikey")
        return
    
    # Create the MCP client (connects to localhost:8000 by default)
    host = os.getenv("MCP_SERVER_HOST", "localhost")
    port = os.getenv("MCP_SERVER_PORT", "8000")
    mcp_client = MCPClient(f"http://{host}:{port}")
    
    # Make sure the server is running
    if not await check_server_health(mcp_client):
        return
    
    # Pre-load the tools (caches them for later)
    print("Loading available tools...", end=" ")
    try:
        tools = await mcp_client.list_tools()
        print(f"✅ Found {len(tools)} tools")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return
    
    # Show the welcome message
    print_header()
    
    # Main loop - keep taking questions until user quits
    while True:
        try:
            # Get input from user
            question = input("You: ").strip()
            
            # Skip empty input
            if not question:
                continue
            
            # Check for exit commands
            if question.lower() in ("quit", "exit", "q"):
                print("\nGoodbye! 👋\n")
                break
            
            # Process the question
            await process_question(question, mcp_client)
            
        except KeyboardInterrupt:
            # User pressed Ctrl+C
            print("\n\nGoodbye! 👋\n")
            break
        except EOFError:
            # Input stream ended (e.g., piped input)
            print("\nGoodbye! 👋\n")
            break


def main():
    """
    Entry point for the CLI.
    
    Sets up minimal logging (we don't want to spam the console)
    and runs the async main loop.
    """
    # Keep logging quiet - only show warnings and errors
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    
    # Run the async main loop
    asyncio.run(main_loop())


if __name__ == "__main__":
    main()
