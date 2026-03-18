from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",  # vLLM 不校验 key，随便填
    base_url="http://localhost:8012/v1"
)

# 计时
import time
start_time = time.time()
print("=== Sending Request to Qwen2.5-VL-7B-Instruct ===")
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-4B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "请用一句话解释什么是大语言模型。"}
    ],
    temperature=0.7,
    max_tokens=12800,
)
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

print("=== Model Output ===")
print(response.choices[0].message.content)
