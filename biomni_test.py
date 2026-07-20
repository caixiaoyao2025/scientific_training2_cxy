import os

os.environ["MCP_DATA_ROOT"] = os.path.join(os.path.dirname(__file__), "data")
os.environ["MCP_APP_ROOT"] = os.path.dirname(__file__)

from biomni.agent import A1

agent = A1(
    path=os.path.join(os.path.dirname(__file__), "data"),
    llm="qwen2.5:14b",
    source="Custom",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    expected_data_lake_files=[],
)
agent.add_mcp(config_path="./mcp_config_cluster.yaml")

result = agent.go("Use list_registered_tools tool to show all available tools")
print(result)
