"""FastAPI collector for Tok telemetry events."""

import sqlite3
import json
import os
from typing import Any

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="Tok Telemetry Collector")

DB_PATH = os.getenv("TOK_COLLECTOR_DB", "telemetry.db")


class TokEvent(BaseModel):
    event_type: str
    timestamp: str
    request_id: str
    model: str
    payload: dict[str, Any]


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            timestamp TEXT,
            request_id TEXT,
            model TEXT,
            payload_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


@app.on_event("startup")
async def startup_event() -> None:
    init_db()


def save_event(event: TokEvent) -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO events (event_type, timestamp, request_id, model, payload_json) VALUES (?, ?, ?, ?, ?)",
        (
            event.event_type,
            event.timestamp,
            event.request_id,
            event.model,
            json.dumps(event.payload),
        ),
    )
    conn.commit()
    conn.close()


@app.post("/ingest")
async def ingest_event(
    event: TokEvent, background_tasks: BackgroundTasks
) -> dict[str, str]:
    background_tasks.add_task(save_event, event)
    return {"status": "accepted"}


@app.get("/events")
async def get_events(limit: int = 100) -> list[dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
