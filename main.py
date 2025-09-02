#!/usr/bin/env python3
"""
OpenAI Chatbot API
This script exposes an API endpoint for the OpenAI chatbot using FastAPI
"""

import os
import openai
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create FastAPI app
app = FastAPI()

# Data model for the chat request
class ChatRequest(BaseModel):
    message: str

# Chat endpoint that uses OpenAI
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": request.message}]
    )
    return {"reply": response.choices[0].message['content']}

# Run the app using Uvicorn if executed as main
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
