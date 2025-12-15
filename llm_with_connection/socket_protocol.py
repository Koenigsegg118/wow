import numpy as np


class StateReceiver:
    """
    AFSIM 发来的文本帧格式：
      simTime 640 v0 v1 ... v639
    由于 TCP 可能半包/粘包，这里用 token 缓冲，凑够一帧(2+state_len 个 token)再解析。
    """

    def __init__(self):
        self._text_buf = ""
        self._tokens: list[str] = []

    def _feed(self, conn) -> None:
        data = conn.recv(8192)
        if not data:
            raise EOFError("socket closed")

        s = data.decode("ascii", errors="ignore")
        self._text_buf += s

        if self._text_buf and (not self._text_buf[-1].isspace()):
            parts = self._text_buf.split()
            if parts:
                self._text_buf = parts[-1]
                self._tokens.extend(parts[:-1])
            return

        parts = self._text_buf.split()
        self._text_buf = ""
        self._tokens.extend(parts)

    def recv_frame(self, conn):
        while True:
            while len(self._tokens) < 2:
                self._feed(conn)

            state_len = int(float(self._tokens[1]))
            need = 2 + state_len

            while len(self._tokens) < need:
                self._feed(conn)

            frame = self._tokens[:need]
            del self._tokens[:need]

            sim_time = float(frame[0])
            vals = np.asarray(frame[2 : 2 + state_len], dtype=np.float32)
            return sim_time, vals


def send_status_data(connection, action_640_f32: np.ndarray) -> None:
    action_640_f32 = np.asarray(action_640_f32, dtype=np.float32)
    if action_640_f32.size != 640:
        raise ValueError(f"action size must be 640 float32, got {action_640_f32.size}")

    connection.sendall(b"STATUS")
    payload = action_640_f32.astype("<f4", copy=False).tobytes()
    connection.sendall(payload)


def send_reset_instruction(connection) -> None:
    connection.sendall(b"RESET")

