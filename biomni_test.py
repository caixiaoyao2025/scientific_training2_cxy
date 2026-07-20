from biomni.agent import A1

agent = A1(
    path='D:/bio-data',
    llm='qwen2.5:14b',
    source='Custom',
    base_url='http://localhost:11434/v1',
    api_key='ollama',
    expected_data_lake_files=[],
)
agent.add_mcp(config_path="./mcp_config.yaml")

result = agent.go("Use list_registered_tools tool to show all available tools")
print(result)
