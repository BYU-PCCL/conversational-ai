#!/usr/bin/env python3
"""Simnple chatbot websocket based server."""

import json

from sanic import Sanic, request, response
from sanic.websocket import WebSocketProtocol  # noqa

from chat import chatbot

app = Sanic(name="conversational-ai")

app.static("/", "/index.html")


@app.route("/chats/<convo_id>")
async def _chats(request: request, convo_id: str) -> response:
    convo = request.app.chat_histories.get(
        convo_id, "[no chat history for requested convo_id]"
    )
    return response.text(convo)


@app.route("/chats/delete/<convo_id>")
async def _chats_delete(request: request, convo_id: str) -> response:
    _ = request.app.chat_histories.pop(convo_id, "")
    return response.text("deleted all chat history for: {}".format(convo_id))


@app.websocket("/chat")
async def _chat(request: request, ws: Sanic.websocket) -> None:
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
    app.run(host="localhost", port=8080, protocol=WebSocketProtocol)
