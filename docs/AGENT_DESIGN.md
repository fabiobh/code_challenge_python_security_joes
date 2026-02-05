# LangGraph Agent Design

This document explains the internal design of the ReAct agent, focusing on state management, graph flow, and the tool-calling protocol.

---

## LangGraph State

The agent's state is the "memory" passed between nodes. Defined in [state.py](file:///c:/Users/fabio/OneDrive/Documents/GitHub/code_challenge_python_security_joes/src/agent/state.py):

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history (auto-appends)
    tool_calls: list[ToolCall]               # Trace of all executed tools
    current_step: str                        # "think", "act", or "done"
    final_answer: Optional[str]              # The response to return
    pending_tool_call: Optional[dict]        # Tool queued for execution
```

| Field | Purpose |
|-------|---------|
| `messages` | Stores user question, LLM responses, and tool results. Uses LangGraph's `add_messages` reducer to auto-append. |
| `tool_calls` | Audit log of every tool invocation (name, params, result, error). Displayed at the end. |
| `current_step` | Controls graph routing: `think` → LLM decides, `act` → execute tool, `done` → exit. |
| `final_answer` | Set when LLM returns `{"answer": "..."}`. Becomes the user-facing response. |
| `pending_tool_call` | Holds `{"tool": "...", "params": {...}}` between Think and Act nodes. |

---

## Nodes & Graph Flow

The graph implements a **Think → Act → Observe** loop:

```
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
```

### Think Node
1. Sends `SystemPrompt + messages` to Gemini LLM.
2. Parses JSON response (cleans markdown code fences if present).
3. Updates state:
   - `{"tool": ...}` → sets `pending_tool_call`, `current_step="act"`
   - `{"answer": ...}` → sets `final_answer`, `current_step="done"`
   - Parse failure → treats raw text as `final_answer`

### Act Node
1. Reads `pending_tool_call` from state.
2. Calls MCP server via `MCPClient.call_tool()`.
3. Appends result (or error) to `messages` as a `HumanMessage`.
4. Records call in `tool_calls` trace.
5. Sets `current_step="think"` to return to LLM.

### Routing
Conditional edge after Think:
- `current_step == "done"` → `END`
- `current_step == "act"` → Act node
- Fallback → `END`

---

## Tool-Calling Strategy

### Prompt-Based JSON Protocol
The agent uses **prompt-based tool calling**, not native function calling. The LLM is instructed to output JSON:

```json
{"tool": "ipify", "params": {}}
```
or
```json
{"answer": "The weather is..."}
```

### Enforced Tool Order
The system prompt mandates a specific sequence:
1. **ipify** → Get public IP of data center
2. **ip_to_geo** → Convert IP to latitude/longitude
3. **weather_forecast** → Fetch weather for coordinates

The LLM chains tool outputs: each result informs the next call's parameters.