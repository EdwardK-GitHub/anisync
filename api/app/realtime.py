import datetime as dt
from collections import defaultdict

from fastapi import WebSocket


class RoomConnectionManager:
    """
    In-memory WebSocket connection manager.

    This is intentionally ephemeral.
    PostgreSQL remains the authoritative source of room state.
    """

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, room_code: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[room_code].add(websocket)

    def disconnect(self, room_code: str, websocket: WebSocket) -> None:
        self._connections[room_code].discard(websocket)
        if not self._connections[room_code]:
            self._connections.pop(room_code, None)

    async def broadcast(
        self,
        room_code: str,
        *,
        event_type: str,
        state_revision: int,
        changed_sections: list[str],
    ) -> None:
        """
        Broadcast a lightweight event envelope.

        Clients should re-fetch authoritative JSON sections over HTTP.
        """
        payload = {
            "event_type": event_type,
            "room_code": room_code,
            "state_revision": state_revision,
            "changed_sections": changed_sections,
            "server_timestamp": dt.datetime.now(dt.UTC).isoformat(),
        }

        dead_connections: list[WebSocket] = []
        for websocket in list(self._connections.get(room_code, set())):
            try:
                await websocket.send_json(payload)
            except Exception:
                dead_connections.append(websocket)

        for websocket in dead_connections:
            self.disconnect(room_code, websocket)


manager = RoomConnectionManager()
