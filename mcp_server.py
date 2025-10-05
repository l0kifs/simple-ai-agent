from fastmcp import FastMCP

mcp_server = FastMCP("Калькулятор и утилиты")


@mcp_server.tool()
def add(a: float, b: float) -> float:
    """Складывает два числа."""
    return a + b


@mcp_server.tool()
def multiply(a: float, b: float) -> float:
    """Умножает два числа."""
    return a * b


@mcp_server.tool()
def get_weather(city: str) -> dict:
    """
    Получает информацию о погоде для указанного города.
    (Это симуляция - в реальном приложении здесь был бы API запрос)
    """
    return {"city": city, "temperature": 22, "condition": "Солнечно", "humidity": 65}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        mcp_server.run()
