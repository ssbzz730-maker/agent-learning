"""第8课FastAPI：真实LangGraph事件流、心跳与客户端断线取消。"""

import asyncio
import json
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from lesson_07_fastapi_deployment.api import (
    AgentResponse,
    ApprovalRequest,
    ChatRequest,
)
from lesson_08_streaming_agent.service import default_streaming_service


STATIC_DIR = Path(__file__).parent / "static"
STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
STREAM_END = object()


def sse(event, data):
    """把事件名和JSON数据编码成SSE文本块。"""

    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


async def event_response(request, source, heartbeat_seconds=15):
    """桥接Agent事件和HTTP响应；断线时取消生产任务。"""

    queue = asyncio.Queue()

    async def produce():
        try:
            async for item in source:
                await queue.put(item)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await queue.put(
                {
                    "event": "error",
                    "data": {
                        "type": type(error).__name__,
                        "message": "Agent请求失败",
                    },
                }
            )
        finally:
            await queue.put(STREAM_END)

    producer = asyncio.create_task(produce())
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                item = await asyncio.wait_for(
                    queue.get(),
                    timeout=heartbeat_seconds,
                )
            except TimeoutError:
                yield sse("heartbeat", {"status": "running"})
                continue
            if item is STREAM_END:
                break
            yield sse(item["event"], item["data"])
    finally:
        if not producer.done():
            producer.cancel()
        with suppress(asyncio.CancelledError):
            await producer


async def final_payload(source):
    """为普通REST兼容接口消费事件流，并取出最终result事件。"""

    result = None
    async for item in source:
        if item["event"] == "result":
            result = item["data"]
    if result is None:
        raise RuntimeError("Agent没有返回最终结果")
    return result


def create_app(service=None):
    """创建第8课应用；测试可注入异步假服务。"""

    @asynccontextmanager
    async def lifespan(app):
        if service is not None:
            app.state.agent_service = service
            yield
            return
        async with default_streaming_service() as real_service:
            app.state.agent_service = real_service
            yield

    app = FastAPI(
        title="流式企业知识与工单Agent",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def get_service(request: Request):
        return request.app.state.agent_service

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "streaming-agent-api", "version": "2.0.0"}

    @app.post("/api/chat", response_model=AgentResponse)
    async def chat(body: ChatRequest, agent_service=Depends(get_service)):
        try:
            return await final_payload(
                agent_service.stream_chat(
                    body.question,
                    body.thread_id,
                    body.max_steps,
                )
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail="Agent暂时不可用") from error

    @app.post("/api/chat/stream")
    async def chat_stream(
        body: ChatRequest,
        request: Request,
        agent_service=Depends(get_service),
    ):
        source = agent_service.stream_chat(
            body.question,
            body.thread_id,
            body.max_steps,
        )
        return StreamingResponse(
            event_response(request, source),
            media_type="text/event-stream",
            headers=STREAM_HEADERS,
        )

    @app.post("/api/approvals", response_model=AgentResponse)
    async def approval(body: ApprovalRequest, agent_service=Depends(get_service)):
        try:
            return await final_payload(
                agent_service.stream_resume(body.thread_id, body.approved)
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail="该会话无法恢复") from error

    @app.post("/api/approvals/stream")
    async def approval_stream(
        body: ApprovalRequest,
        request: Request,
        agent_service=Depends(get_service),
    ):
        source = agent_service.stream_resume(body.thread_id, body.approved)
        return StreamingResponse(
            event_response(request, source),
            media_type="text/event-stream",
            headers=STREAM_HEADERS,
        )

    @app.get("/api/sessions/{thread_id}")
    async def session(thread_id: str, agent_service=Depends(get_service)):
        try:
            return await agent_service.get_session(thread_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="会话不存在") from error

    return app


app = create_app()
