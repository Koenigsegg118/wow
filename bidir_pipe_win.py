import struct
import time

PIPE_NAME = r"\\.\pipe\cpp_py_pipe"


def read_exact(f, n: int) -> bytes | None:
    """从文件对象 f 中精确读取 n 字节，读不到返回 None"""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = f.read(remaining)
        if not chunk:
            return None  # EOF
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(f):
    """读一条消息：4字节小端长度 + payload"""
    header = read_exact(f, 4)
    if header is None:
        return None
    (length,) = struct.unpack("<I", header)  # 小端 uint32
    if length == 0:
        return b""
    payload = read_exact(f, length)
    if payload is None:
        return None
    return payload


def send_message(f, payload: bytes):
    """写一条消息：4字节小端长度 + payload"""
    header = struct.pack("<I", len(payload))
    f.write(header)
    if payload:
        f.write(payload)
    f.flush()


def main():
    print("Python: waiting to connect to pipe:", PIPE_NAME)
    # 这里 open 会阻塞直到 C++ 那边 CreateNamedPipe+ConnectNamedPipe 完成
    f = open(PIPE_NAME, "r+b", buffering=0)
    print("Python: connected to pipe, entering loop.")

    while True:
        payload = recv_message(f)
        if payload is None:
            print("Python: EOF or error on recv, exit.")
            break

        float_count = len(payload) // 4
        floats = list(struct.unpack(f"<{float_count}f", payload))
        print(f"Python: received {float_count} floats, first 3 = {floats[:3]}")

        # 示例处理：把每个 float * 0.5 再回传
        processed = [x * 0.5 for x in floats]
        reply_payload = struct.pack(f"<{len(processed)}f", *processed)

        send_message(f, reply_payload)
        print(f"Python: sent {len(processed)} floats back.\n")

        # 根据需要可以 sleep，这里不强制
        # time.sleep(0.1)

    f.close()
    print("Python: done.")


if __name__ == "__main__":
    main()