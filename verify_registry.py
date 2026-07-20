import asyncio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

async def test():
    params = StdioServerParameters(
        command="docker",
        args=["run", "--rm", "-i", "-v", "C:\\Users\\123456\\Desktop\\st2\\SCI1003_TEAM11\\data:/data", "bio-mcp"],
    )
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"Found {len(tools.tools)} tools:\n")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description[:80]}")

asyncio.run(test())
