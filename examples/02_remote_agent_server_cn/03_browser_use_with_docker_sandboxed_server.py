import os
import platform
import time

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


def detect_platform():
    """检测合适的 Docker platform 字符串。"""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"


# 创建一个基于 Docker 的远程工作区，并开放额外端口以便浏览器访问
with DockerWorkspace(
    base_image="nikolaik/python-nodejs:python3.12-nodejs22",
    host_port=8010,
    # TODO: 如果不是 linux/arm64，请根据实际情况调整 platform
    platform=detect_platform(),
    extra_ports=True,  # 暴露额外端口以用于 VSCode 与 VNC
    forward_env=["LLM_API_KEY"],  # 将 API 密钥转发至容器
) as workspace:
    """设置 extra_ports=True 后，可通过 localhost:8012 访问 VNC"""

    # 创建启用浏览器工具的智能体
    agent = get_default_agent(
        llm=llm,
        cli_mode=False,  # 将 CLI 模式设为 False 以启用浏览器工具
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
    conversation.send_message(
        "请访问 https://all-hands.dev/ 的博客页面，总结最新一篇博客的要点。"
    )
    conversation.run()

    # 等待用户确认后退出
    y = None
    while y != "y":
        y = input(
            "由于在 DockerWorkspace 中启用了 extra_ports=True，"
            "可以打开浏览器标签查看 OpenHands 通过 VNC 控制的真实浏览器。\n\n"
            "链接: http://localhost:8012/vnc.html?autoconnect=1&resize=remote\n\n"
            "按下 'y' 再回车以退出并终止工作区。\n"
            ">> "
        )
