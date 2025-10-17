import os
import platform
import time

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Conversation,
    RemoteConversation,
    get_logger,
)
from openhands.tools.preset.default import get_default_agent
from openhands.workspace import DockerWorkspace


logger = get_logger(__name__)


# 1）确保存在 LLM API 密钥
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


# 2）创建一个基于 Docker 的远程工作区，会自动构建并管理容器
with DockerWorkspace(
    # 动态构建 agent-server 镜像
    # base_image="nikolaik/python-nodejs:python3.12-nodejs22",
    # 使用预构建镜像以加快启动速度
    server_image="ghcr.io/all-hands-ai/agent-server:latest-python",
    host_port=8010,
    platform=detect_platform(),
    forward_env=["LLM_API_KEY"],  # 将 API 密钥转发到容器内
) as workspace:
    # 3）创建智能体
    agent = get_default_agent(
        llm=llm,
        cli_mode=True,
    )

    # 4）设置回调收集
    received_events: list = []
    last_event_time = {"ts": time.time()}

    def event_callback(event) -> None:
        event_type = type(event).__name__
        logger.info(f"🔔 回调收到事件: {event_type}\n{event}")
        received_events.append(event)
        last_event_time["ts"] = time.time()

    # 5）通过一个简单命令测试工作区
    result = workspace.execute_command(
        "echo '来自沙箱环境的问候！' && pwd"
    )
    logger.info(
        f"命令 '{result.command}' 执行完成，退出码 {result.exit_code}"
    )
    logger.info(f"输出: {result.stdout}")
    conversation = Conversation(
        agent=agent,
        workspace=workspace,
        callbacks=[event_callback],
        visualize=True,
    )
    assert isinstance(conversation, RemoteConversation)

    try:
        logger.info(f"\n📋 对话 ID: {conversation.state.id}")

        logger.info("📝 发送第一条消息……")
        conversation.send_message(
            "阅读当前仓库，并将与项目相关的 3 条事实写入 FACTS.txt。"
        )
        logger.info("🚀 正在运行对话……")
        conversation.run()
        logger.info("✅ 第一项任务已完成！")
        logger.info(f"智能体状态: {conversation.state.agent_status}")

        # 等待事件稳定（2 秒内无事件）
        logger.info("⏳ 正在等待事件停止……")
        while time.time() - last_event_time["ts"] < 2.0:
            time.sleep(0.1)
        logger.info("✅ 事件已停止")

        logger.info("🚀 再次运行对话……")
        conversation.send_message("很好！现在请删除那个文件。")
        conversation.run()
        logger.info("✅ 第二项任务已完成！")
    finally:
        print("\n🧹 正在清理由话……")
        conversation.close()
