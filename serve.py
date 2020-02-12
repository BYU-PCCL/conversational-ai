#!/usr/bin/env python3

import json
import os
from pathlib import Path

from sanic import Sanic, response
from sanic.websocket import WebSocketProtocol

from chat import chatbot

app = Sanic(name="conversational-ai")

app.static("/", "/index.html")


@app.route("/chats/<convo_id>")
async def _chats(request, convo_id):
    convo = request.app.chat_histories.get(convo_id, "[no chat history for requested convo_id]")
    return response.text(convo)


@app.route("/chats/delete/<convo_id>")
async def _chats_delete(request, convo_id):
    _ = request.app.chat_histories.pop(convo_id, "")
    return response.text("deleted all chat history for: {}".format(convo_id))


@app.websocket("/chat")
async def _chat(request, ws):
    while True:
        data = await ws.recv()
        data = json.loads(data)
        print("received:", data)

        user_input = data.get("message")
        response = ""
        if user_input:
            convo = request.app.chat_histories.setdefault(data["id"], "")
            response, convo = request.app.chat(user_input, conversation=convo)
            request.app.chat_histories[data["id"]] = convo

        print("sending:", response)
        await ws.send(response)


if __name__ == "__main__":
    # TODO: add support for saving chat history to db/fs
    # app state
    app.chat = chatbot(checkpoint="checkpoint/")
    app.chat_histories = {}
    app.run(host="0.0.0.0", port=8080, protocol=WebSocketProtocol)
