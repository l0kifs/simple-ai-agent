# simple-ai-agent

A simple AI agent for educational purposes, demonstrating integration with Model Context Protocol (MCP) and OpenRouter API.

## Description

The project implements an AI agent that can interact with language models via the OpenRouter API and use tools provided by MCP servers. The agent supports conversation context management, calling built-in tools, and connecting external MCP servers.

## Features

- **Built-in tools**:
  - Get current date and time
  - Check the number of tokens in the context
  - Clear the conversation context

- **MCP tools** (via connected server):
  - Mathematical operations (addition, multiplication)
  - Get weather information (simulation)

- **LLM Integration**: Support for models via OpenRouter API (e.g., DeepSeek)

## Setup

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure API key**:
   - Create a `.env` file in the project root
   - Add your OpenRouter API key:
     ```
     OPENROUTER_API_KEY=your_api_key_here
     ```
   - Get the key at [openrouter.ai](https://openrouter.ai)

## Running

Run the agent with examples:
```bash
uv run python main.py
```

The agent will automatically connect to the MCP server and execute demonstration queries.

## Project Structure

- `ai_agent.py` - Main agent class and built-in tools
- `mcp_server.py` - MCP server with mathematical and weather tools
- `main.py` - Entry point with usage examples
- `settings.py` - Configuration (API keys and parameters)
- `pyproject.toml` - Project dependencies and metadata
