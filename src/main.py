from fastapi import FastAPI

app = FastAPI(title="Jarvis Core", version="0.1.0")

@app.get("/health")
async def health_check():
    """This endpoint lets us know the brain is online."""
    return {"status": "online", "system": "Jarvis"}