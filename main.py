import asyncio
import signal

from ai_agent import Agent, AgentSettings
from settings import settings as general_settings


async def main():
    settings = AgentSettings(openrouter_api_key=general_settings.openrouter_api_key)
    print(f"Max tool call iterations: {settings.max_tool_call_iterations}")
    print(f"Max context tokens: {settings.max_context_tokens}")

    def signal_handler(signum, frame):
        print(f"Received signal {signum}. Shutting down gracefully...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    async with Agent(settings=settings) as agent:
        try:
            # Connect MCP server
            await agent.add_mcp_stdio_server("python", ["mcp_server.py", "--server"])

            agent.set_system_message(
                "You are a smart assistant. Answer briefly and to the point."
            )
            print(">>> ", await agent.send_message("My favorite color is blue."))
            print(">>> ", await agent.send_message("What is my favorite color?"))
            # print(">>> ", await agent.send_message("What time is it?"))
            print(
                ">>> ", await agent.send_message("How many tokens are in the context?")
            )
            print(">>> ", await agent.send_message("Clear the context."))
            print(
                ">>> ", await agent.send_message("How many tokens are in the context?")
            )
            print("\n=== Agent Context ===")
            for i, msg in enumerate(agent._context):
                print(f"{i + 1}. {msg['role']}: {msg['content'][:100]}...")

            # Example for checking tool call loop
            print("\n=== Tool Call Loop Example ===")
            agent.reset_context()
            agent.set_system_message(
                "You are a smart assistant. Answer briefly. If the user asks for the time, first get the time with a tool, then check the number of tokens, and if tokens are more than 50, clear the context."
            )
            response = await agent.send_message("What time is it?")
            print("Agent response:", response)
            print("Context after response:")
            for i, msg in enumerate(agent._context):
                print(f"{i + 1}. {msg['role']}: {msg['content'][:50]}...")

            # Test MCP tools
            print("Available tools:", list(agent._tools.keys()))
            response = await agent.send_message("Add 15 and 27")
            print("Agent response:", response)

        except KeyboardInterrupt:
            print("Interrupt received. Shutting down the agent...")
        except Exception as e:
            print(f"An error occurred: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
