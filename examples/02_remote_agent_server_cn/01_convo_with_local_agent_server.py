import os
import subprocess
import sys
import threading
import time

from pydantic import SecretStr

from openhands.sdk import LLM, Conversation, RemoteConversation, Workspace, get_logger
from openhands.sdk.event import ConversationStateUpdateEvent
from openhands.tools.preset.default import get_default_agent


logger = get_logger(__name__)


def _stream_output(stream, prefix, target_stream):
    """将子进程输出带前缀地写入目标输出流。"""
    try:
        for line in iter(stream.readline, ""):
            if line:
                target_stream.write(f"[{prefix}] {line}")
                target_stream.flush()
    except Exception as e:
        print(f"转发 {prefix} 输出时出错: {e}", file=sys.stderr)
    finally:
        stream.close()


class ManagedAPIServer:
    """通过上下文管理器控制 OpenHands API 服务器子进程。"""

    def __init__(self, port: int = 8000, host: str = "127.0.0.1"):
        self.port = port
        self.host = host
        self.process = None
        self.base_url = f"http://{host}:{port}"
        self.stdout_thread = None
        self.stderr_thread = None

    def __enter__(self):
        """启动 API 服务器子进程。"""
        print(f"正在 {self.base_url} 启动 OpenHands API 服务器……")

        # 启动服务器进程
        self.process = subprocess.Popen(
            [
                "python",
                "-m",
                "openhands.agent_server",
                "--port",
                str(self.port),
                "--host",
                self.host,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"LOG_JSON": "true", **os.environ},
        )

        # 分别启动线程转发标准输出与错误输出
        self.stdout_thread = threading.Thread(
            target=_stream_output,
            args=(self.process.stdout, "SERVER", sys.stdout),
            daemon=True,
        )
        self.stderr_thread = threading.Thread(
            target=_stream_output,
            args=(self.process.stderr, "SERVER", sys.stderr),
            daemon=True,
        )

        self.stdout_thread.start()
        self.stderr_thread.start()

        # 等待服务器就绪
        max_retries = 30
        for _ in range(max_retries):
            try:
                import httpx

                response = httpx.get(f"{self.base_url}/health", timeout=1.0)
                if response.status_code == 200:
                    print(f"API 服务器已在 {self.base_url} 就绪")
                    return self
            except Exception:
                pass

            if self.process.poll() is not None:
                # 进程已经退出
                raise RuntimeError(
                    "服务器进程意外终止。"
                    "请查看上方服务器日志以获取详细信息。"
                )

            time.sleep(1)

        raise RuntimeError(f"服务器在 {max_retries} 秒后仍未启动")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """停止 API 服务器子进程。"""
        if self.process:
            print("正在停止 API 服务器……")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("终止超时，正在强制杀死 API 服务器……")
                self.process.kill()
                self.process.wait()

            # 等待转发线程结束（daemon 线程会自动退出），但留一点时间冲刷剩余输出
            time.sleep(0.5)
            print("API 服务器已停止。")


api_key = os.getenv("LLM_API_KEY")
assert api_key is not None, "尚未设置 LLM_API_KEY 环境变量。"

llm = LLM(
    service_id="agent",
    model="litellm_proxy/anthropic/claude-sonnet-4-5-20250929",
    base_url="https://llm-proxy.eval.all-hands.dev",
    api_key=SecretStr(api_key),
)
title_gen_llm = LLM(
    service_id="title-gen-llm",
    model="litellm_proxy/openai/gpt-5-mini",
    base_url="https://llm-proxy.eval.all-hands.dev",
    api_key=SecretStr(api_key),
)

# 使用托管的 API 服务器
with ManagedAPIServer(port=8001) as server:
    # 创建智能体
    agent = get_default_agent(
        llm=llm,
        cli_mode=True,  # 为了简化示例，禁用浏览器工具
    )

    # 定义回调用于验证 WebSocket 功能
    received_events = []
    event_tracker = {"last_event_time": time.time()}

    def event_callback(event):
        """用于捕获事件的回调函数。"""
        event_type = type(event).__name__
        logger.info(f"🔔 回调收到事件: {event_type}\n{event}")
        received_events.append(event)
        event_tracker["last_event_time"] = time.time()

    # 创建带回调的远程对话
    # 注意：RemoteConversation 需要 Workspace 实例
    workspace = Workspace(host=server.base_url)
    result = workspace.execute_command("pwd")
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

        # 发送首条消息并开始运行
        logger.info("📝 发送第一条消息……")
        conversation.send_message(
            "阅读当前仓库，并将与项目相关的 3 条事实写入 FACTS.txt。"
        )

        # 使用指定 LLM 生成标题
        title = conversation.generate_title(max_length=60, llm=title_gen_llm)
        logger.info(f"生成的对话标题: {title}")

        logger.info("🚀 正在运行对话……")
        conversation.run()

        logger.info("✅ 第一项任务已完成！")
        logger.info(f"智能体状态: {conversation.state.agent_status}")

        # 等待事件停止（2 秒内无事件）
        logger.info("⏳ 正在等待事件停止……")
        while time.time() - event_tracker["last_event_time"] < 2.0:
            time.sleep(0.1)
        logger.info("✅ 事件已停止")

        logger.info("🚀 再次运行对话……")
        conversation.send_message("很好！现在请删除那个文件。")
        conversation.run()
        logger.info("✅ 第二项任务已完成！")

        # 展示 state.events 的用法
        logger.info("\n" + "=" * 50)
        logger.info("📊 演示状态事件 API")
        logger.info("=" * 50)

        # 使用 state.events 统计事件总数
        total_events = len(conversation.state.events)
        logger.info(f"📈 对话中的事件总数: {total_events}")

        # 使用 state.events 获取最近 5 个事件
        logger.info("\n🔍 使用 state.events 获取最近 5 个事件……")
        all_events = conversation.state.events
        recent_events = all_events[-5:] if len(all_events) >= 5 else all_events

        for i, event in enumerate(recent_events, 1):
            event_type = type(event).__name__
            timestamp = getattr(event, "timestamp", "未知时间")
            logger.info(f"  {i}. {event_type} @ {timestamp}")

        # 查看有哪些事件类型
        logger.info("\n🔍 已发现的事件类型:")
        event_types = set()
        for event in recent_events:
            event_type = type(event).__name__
            event_types.add(event_type)
        for event_type in sorted(event_types):
            logger.info(f"  - {event_type}")

        # 输出所有 ConversationStateUpdateEvent
        logger.info("\n🗂️  ConversationStateUpdateEvent 事件列表:")
        for event in conversation.state.events:
            if isinstance(event, ConversationStateUpdateEvent):
                logger.info(f"  - {event}")

    finally:
        # 清理资源
        print("\n🧹 正在清理由话……")
        conversation.close()
