class QwenCodeLLMServer:
    async def generate_response(self, messages, max_tokens=4096, temperature=0.1):
        # 실제 LLM API 연동 필요. 아래는 데모용
        return '{"content": "이곳에 LLM 응답이 들어갑니다."}' 