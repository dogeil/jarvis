from pydantic import BaseModel
from typing import Optional, Any

class JarvisMessage(BaseModel):
    source: str          # e.g., "hand_engine", "esp32", "voice"
    type: str            # e.g., "gesture", "sensor_reading", "command"
    payload: Any         # The actual data
    timestamp: float