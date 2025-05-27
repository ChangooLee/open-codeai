import requests

url = "http://127.0.0.1:8003/api/agent/chat"
data = {"message": "src/agent_tools.py 파일을 읽어줘"}
response = requests.post(url, json=data)
print(response.json()) 