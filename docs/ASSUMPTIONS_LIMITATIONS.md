## Assumptions & Limitations

### Assumptions
| Assumption | Implication |
|------------|-------------|
| "Data center" = MCP server machine | Weather reflects where the server runs, not the user. |
| Public IP ≈ geographic location | May be inaccurate for VPNs, proxies, or cloud providers. |
| `GOOGLE_API_KEY` is set | Agent fails immediately without it. |
| MCP server at `localhost:8000` | Configurable via `MCP_SERVER_HOST`/`MCP_SERVER_PORT`. |

### Limitations
| Limitation | Details |
|------------|---------|
| **Sync/Async bridging** | `act_node_sync` wraps async HTTP calls in `loop.run_until_complete()`. Uses `nest_asyncio` to allow nested loops. |
| **Empty LLM responses** | Retries up to 3 times. Falls back to constructing answer from collected tool data if available. |
| **No conversation persistence** | State resets on each question. No multi-turn memory. |
| **Single-threaded** | Processes one question at a time. |
| **Rate limits** | IP-API: 45 req/min. Gemini free tier: 15 RPM. |
| **IP-API HTTP-only** | Free tier doesn't support HTTPS. |

### Error Handling
- MCP errors (`MCPToolError`, `MCPClientError`) are caught and added to `messages`.
- LLM sees error text and can decide to retry or respond with failure.
- JSON parse failures treat raw text as `final_answer` (graceful degradation).
