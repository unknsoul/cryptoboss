"""Quick WebSocket test."""
import asyncio
import websockets
import json

async def test():
    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        for i in range(3):
            msg = await asyncio.wait_for(ws.recv(), timeout=10)
            data = json.loads(msg)
            msg_type = data.get("type", "?")
            price = data.get("data", {}).get("price", "?")
            symbol = data.get("data", {}).get("symbol", "?")
            print(f"WS msg {i+1}: type={msg_type}, symbol={symbol}, price={price}")
        print("WebSocket OK!")

asyncio.run(test())
