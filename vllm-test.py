from openai import OpenAI

client = OpenAI(
    base_url="http://195.242.10.142:8000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="google/gemma-4-E4B-it",
    messages=[
        {"role": "user", "content": "Write a poem about the ocean."}
    ],
    max_tokens=512,
    temperature=0.7
)

print(response.choices[0].message.content)
