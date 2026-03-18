from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json

app = FastAPI(title="Jarvis Core")


@app.get("/health")
async def health_check():
    """
    Lightweight health endpoint used by tests and external monitors.
    Returns a simple JSON payload when the API process is alive.
    """
    return {"status": "online", "system": "Jarvis"}


# This list tracks all active "senses" (Hand, Voice, ESP32)
active_connections: list[WebSocket] = []


@app.websocket("/ws/jarvis")
async def jarvis_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Receive data from any core (e.g., Hand Gesture)
            data = await websocket.receive_text()
            message = json.loads(data)

            # Broadcast that data to all other connected cores
            for connection in active_connections:
                if connection != websocket:
                    await connection.send_text(json.dumps(message))
    except WebSocketDisconnect:
        active_connections.remove(websocket)