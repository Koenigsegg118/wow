import requests
from typing import Any


def check_server_connection(base_url: str, server_name: str, timeout: int = 3) -> None:
    """
    测试服务器连接状态。
    尝试访问 /models 端点，这是 OpenAI 兼容接口的标准端点。
    """
    try:
        test_url = f"{base_url.rstrip('/')}/models"
        response = requests.get(test_url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅{server_name} 连接成功!")
            return
        raise ConnectionError(f"服务器响应了，但状态码异常: {response.status_code}")
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"❌无法连接到 {server_name} ({base_url}).\n原因: 目标拒绝连接。请检查 IP/端口/防火墙设置。"
        ) from e
    except requests.exceptions.Timeout as e:
        raise ConnectionError(
            f"❌连接 {server_name} 超时 ({timeout}s).\n原因: 网络延迟过高或服务器无响应。"
        ) from e
    except Exception as e:
        raise ConnectionError(f"❌连接 {server_name} 发生未知错误: {e}") from e


def create_openai_client(
    *,
    api_key: str,
    api_base: str,
    server_name: str,
    timeout: int = 3,
    check_connection: bool = True,
) -> Any:
    try:
        from openai import OpenAI  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'openai'. Please install it (see environment.yml) to use the LLM clients."
        ) from e

    if check_connection:
        print(f"正在连接{server_name}服务器: {api_base}...")
        check_server_connection(api_base, server_name, timeout=timeout)
    return OpenAI(api_key=api_key, base_url=api_base)
