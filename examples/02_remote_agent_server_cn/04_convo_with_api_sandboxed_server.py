"""示例：使用 APIRemoteWorkspace 进行动态构建。

此示例演示如何基于 SDK 代码仓即时构建 agent-server 镜像，
并通过 Runtime API 在远程沙箱环境中启动它。

用法：
  uv run examples/24_remote_convo_with_api_sandboxed_server.py

先决条件：
  - LITELLM_API_KEY：访问 LLM 的 API 密钥
  - RUNTIME_API_KEY：访问 Runtime API 的密钥
"""

import os
import time

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Conversation,
    RemoteConversation,
    get_logger,
)
from openhands.tools.preset.default import get_default_agent
from openhands.workspace import APIRemoteWorkspace


logger = get_logger(__name__)


api_key = os.getenv("LITELLM_API_KEY")
assert api_key, "需要设置 LITELLM_API_KEY"

llm = LLM(
    service_id="agent",
    model="litellm_proxy/anthropic/claude-sonnet-4-5-20250929",
    base_url="https://llm-proxy.eval.all-hands.dev",
    api_key=SecretStr(api_key),
)

runtime_api_key = os.getenv("RUNTIME_API_KEY")
if not runtime_api_key:
    logger.error("需要设置 RUNTIME_API_KEY")
    exit(1)


with APIRemoteWorkspace(
    runtime_api_url="https://runtime.eval.all-hands.dev",
    runtime_api_key=runtime_api_key,
    server_image="ghcr.io/all-hands-ai/agent-server:latest-python",
) as workspace:
    agent = get_default_agent(llm=llm, cli_mode=True)
    received_events: list = []
    last_event_time = {"ts": time.time()}

    def event_callback(event) -> None:
        received_events.append(event)
        last_event_time["ts"] = time.time()

    result = workspace.execute_command(
        "echo '来自沙箱环境的问候！' && pwd"
    )
    logger.info(f"命令执行完成: {result.exit_code}, {result.stdout}")

    conversation = Conversation(
        agent=agent, workspace=workspace, callbacks=[event_callback], visualize=True
    )
    assert isinstance(conversation, RemoteConversation)

    try:
        conversation.send_message(
            "阅读当前仓库，并将与项目相关的 3 条事实写入 FACTS.txt。"
        )
        conversation.run()

        while time.time() - last_event_time["ts"] < 2.0:
            time.sleep(0.1)

        conversation.send_message("很好！现在请删除那个文件。")
        conversation.run()
    finally:
        conversation.close()
