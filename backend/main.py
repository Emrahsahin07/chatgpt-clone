from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    api_key: str
    message: str
    model: str = "gpt-4-1106-preview"

@app.post("/chat")
async def chat(data: ChatRequest):
    client = OpenAI(api_key=data.api_key)

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": data.message}
    ]

    try:
        response = client.chat.completions.create(
            model=data.model,
            messages=messages
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}
