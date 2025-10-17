import os
import time

import httpx
from pydantic import SecretStr

from openhands.sdk import LLM, Conversation, get_logger
from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation
from openhands.tools.preset.default import get_default_agent
from openhands.workspace import DockerWorkspace


logger = get_logger(__name__)


api_key = os.getenv("LLM_API_KEY")
assert api_key is not None, "尚未设置 LLM_API_KEY 环境变量。"

llm = LLM(
    service_id="agent",
    model="litellm_proxy/anthropic/claude-sonnet-4-5-20250929",
    base_url="https://llm-proxy.eval.all-hands.dev",
    api_key=SecretStr(api_key),
)

# 创建一个基于 Docker 的远程工作区，并开放额外端口以便访问 VSCode
with DockerWorkspace(
    base_image="nikolaik/python-nodejs:python3.12-nodejs22",
    host_port=18010,
    # TODO: 如果不是 linux/arm64，请根据实际情况调整 platform
    platform="linux/arm64",
    extra_ports=True,  # 暴露额外端口给 VSCode 与 VNC 使用
    forward_env=["LLM_API_KEY"],  # 将 API 密钥转发至容器
) as workspace:
    """设置 extra_ports=True 后，可通过 localhost:8011 访问 VSCode"""

    # 创建智能体
    agent = get_default_agent(
        llm=llm,
        cli_mode=True,
    )

    # 设置回调收集
    received_events: list = []
    last_event_time = {"ts": time.time()}

    def event_callback(event) -> None:
        event_type = type(event).__name__
        logger.info(f"🔔 回调收到事件: {event_type}\n{event}")
        received_events.append(event)
        last_event_time["ts"] = time.time()

    # 使用该工作区创建远程对话
    conversation = Conversation(
        agent=agent,
        workspace=workspace,
        callbacks=[event_callback],
        visualize=True,
    )
    assert isinstance(conversation, RemoteConversation)

    logger.info(f"\n📋 对话 ID: {conversation.state.id}")
    logger.info("📝 发送第一条消息……")
    conversation.send_message("创建一个输出 Hello World 的简单 Python 脚本")
    conversation.run()

    # 获取带令牌的 VSCode 访问地址
    vscode_port = (workspace.host_port or 8010) + 1
    try:
        response = httpx.get(
            f"{workspace.host}/api/vscode/url",
            params={"workspace_dir": workspace.working_dir},
        )
        vscode_data = response.json()
        vscode_url = vscode_data.get("url", "").replace(
            "localhost:8001", f"localhost:{vscode_port}"
        )
    except Exception:
        # 如果服务器路由不可用，则回退到默认拼接方式
        folder = (
            f"/{workspace.working_dir}"
            if not str(workspace.working_dir).startswith("/")
            else str(workspace.working_dir)
        )
        vscode_url = f"http://localhost:{vscode_port}/?folder={folder}"

    # 等待用户探索 VSCode 后退出
    y = None
    while y != "y":
        y = input(
            "\n"
            "由于在 DockerWorkspace 中启用了 extra_ports=True，"
            "可以打开 VSCode Web 查看该工作区。\n\n"
            f"VSCode 链接: {vscode_url}\n\n"
            "该 VSCode 已预装 OpenHands 设置扩展，并进行了以下配置：\n"
            "  - 启用深色主题\n"
            "  - 启用自动保存\n"
            "  - 禁用遥测\n"
            "  - 禁用自动更新\n\n"
            "按下 'y' 再回车以退出并终止工作区。\n"
            ">> "
        )
