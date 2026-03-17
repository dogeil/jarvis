from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class InteractionLog(Base):
    __tablename__ = "interaction_logs"
    id = Column(Integer, primary_key=True)
    source = Column(String)
    action = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))