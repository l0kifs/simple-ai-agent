import asyncio
import json
from datetime import datetime
from typing import Optional

from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel


class AgentSettings(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = ""
    model_name: str = "deepseek/deepseek-chat-v3.1:free"
    max_tool_call_iterations: int = 5
    max_context_tokens: int = 4000


class GetDatetimeTool:
    name = "get_datetime"
    metadata = {
        "type": "function",
        "function": {
            "name": name,
            "description": "Get the current date and time in ISO 8601 format",
        },
    }

    @classmethod
    async def call(cls, agent: "Agent") -> str:
        return datetime.now().isoformat()


class GetTotalContextTokensTool:
    name = "get_total_context_tokens"
    metadata = {
        "type": "function",
        "function": {
            "name": name,
            "description": "Get the total number of tokens in the current context",
        },
    }

    @classmethod
    async def call(cls, agent: "Agent") -> str:
        return str(agent._total_context_tokens)


class ClearContextTool:
    name = "clear_context"
    metadata = {
        "type": "function",
        "function": {
            "name": name,
            "description": "Clear the current conversation context",
        },
    }

    @classmethod
    async def call(cls, agent: "Agent") -> str:
        agent.reset_context()
        return "Context cleared."


class MCPTool:
    def __init__(self, client: Client, tool_info, keep_alive: bool = False):
        self.client = client
        self.name = tool_info.name
        self.is_persistent = keep_alive
        self.metadata = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": tool_info.description,
                "parameters": tool_info.inputSchema,
            },
        }

    async def call(self, agent: "Agent", **kwargs):
        if self.is_persistent:
            result = await self.client.call_tool(self.name, kwargs)
        else:
            async with self.client:
                result = await self.client.call_tool(self.name, kwargs)
        return result.content[0].text if result.content else ""


class Agent:
    def __init__(self, settings: AgentSettings):
        self._client = AsyncOpenAI(
            base_url=settings.base_url, api_key=settings.openrouter_api_key
        )
        self._model_name = settings.model_name
        self._max_tool_call_iterations = settings.max_tool_call_iterations
        self._max_context_tokens = settings.max_context_tokens
        self._context = []
        self._total_context_tokens = 0
        self._mcp_clients = []
        self._open_clients = []
        self._tools = {
            GetDatetimeTool.name: GetDatetimeTool,
            GetTotalContextTokensTool.name: GetTotalContextTokensTool,
            ClearContextTool.name: ClearContextTool,
        }

    def reset_context(self):
        self._context = []
        self._total_context_tokens = 0

    def trim_context(self):
        """Trim context to stay within max tokens by removing old messages."""
        while (
            self._total_context_tokens > self._max_context_tokens
            and len(self._context) > 1
        ):
            # Remove oldest non-system message
            if self._context[1]["role"] == "system":
                break  # Don't remove system message
            removed = self._context.pop(1)
            # Note: For accurate token counting, use a tokenizer like tiktoken
            # Here we approximate by assuming removal reduces tokens
            self._total_context_tokens -= (
                len(removed.get("content", "")) // 4
            )  # Rough estimate

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        for client in self._open_clients:
            await client.__aexit__(None, None, None)
        self._open_clients = []

    async def add_mcp_stdio_server(
        self,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[dict] = None,
        keep_alive: bool = True,
    ):
        """Add an MCP server via STDIO transport."""
        transport = StdioTransport(
            command=command,
            args=args or [],
            env=env or {},
            keep_alive=keep_alive,
        )
        client = Client(transport)
        self._mcp_clients.append(client)

        if keep_alive:
            await client.__aenter__()
            self._open_clients.append(client)

        async def _load_tools():
            if keep_alive:
                tools = await client.list_tools()
            else:
                async with client:
                    tools = await client.list_tools()
            for tool in tools:
                mcp_tool = MCPTool(client, tool, keep_alive)
                self._tools[mcp_tool.name] = mcp_tool

        await _load_tools()

    async def add_mcp_http_server(
        self, url: str, headers: Optional[dict] = None, keep_alive: bool = True
    ):
        """Add an MCP server via HTTP transport."""
        transport = StreamableHttpTransport(url=url, headers=headers or {})
        client = Client(transport)
        self._mcp_clients.append(client)

        if keep_alive:
            await client.__aenter__()
            self._open_clients.append(client)

        async def _load_tools():
            if keep_alive:
                tools = await client.list_tools()
            else:
                async with client:
                    tools = await client.list_tools()
            for tool in tools:
                mcp_tool = MCPTool(client, tool, keep_alive)
                self._tools[mcp_tool.name] = mcp_tool

        await _load_tools()

    def set_system_message(self, message: str):
        if self._context and self._context[0]["role"] == "system":
            self._context[0]["content"] = message
        else:
            self._context = [{"role": "system", "content": message}] + self._context

    async def _create_chat_completion(self):
        response = await self._client.chat.completions.create(
            model=self._model_name,
            messages=self._context,  # type: ignore
            tools=[tool.metadata for tool in self._tools.values()],  # type: ignore
            tool_choice="auto",  # Let the model decide when to use a tool
        )
        logger.debug("Chat completion response:\n" + response.model_dump_json(indent=2))
        return response

    async def send_message(self, message: str) -> str:
        self._context.append({"role": "user", "content": message})

        iteration = 0
        assistant_message = None
        while iteration < self._max_tool_call_iterations:
            response = await self._create_chat_completion()
            self._total_context_tokens = response.usage.total_tokens
            self.trim_context()

            assistant_message = response.choices[0].message

            assistant_dict = {
                "role": "assistant",
                "content": assistant_message.content or "",
            }
            if assistant_message.tool_calls:
                assistant_dict["tool_calls"] = assistant_message.tool_calls  # type: ignore
            self._context.append(assistant_dict)

            if not assistant_message.tool_calls:
                break

            # Parallel tool calls if independent
            tool_tasks = []
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse tool arguments: {e}")
                    tool_response = f"Error parsing arguments: {str(e)}"
                    self._context.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": tool_response,
                        }
                    )
                    continue

                if function_name in self._tools:
                    tool = self._tools[function_name]
                    # Create task for parallel execution
                    task = self._call_tool_safe(
                        tool, function_args, tool_call.id, function_name
                    )
                    tool_tasks.append(task)
                else:
                    logger.warning(f"Unknown tool: {function_name}")
                    self._context.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": f"Unknown tool: {function_name}",
                        }
                    )

            # Execute all tool calls in parallel
            if tool_tasks:
                await asyncio.gather(*tool_tasks)

            iteration += 1

        return assistant_message.content or "" if assistant_message else ""

    async def _call_tool_safe(self, tool, function_args, tool_call_id, function_name):
        try:
            tool_response = await tool.call(self, **function_args)
        except Exception as e:
            logger.error(f"Tool {function_name} failed: {e}")
            tool_response = f"Error: {str(e)}"
        self._context.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": str(tool_response),
            }
        )
