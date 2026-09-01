"""FastAPI应用工厂、REST接口和SSE事件流。"""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lesson_07_fastapi_deployment.service import build_default_service


STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(min_length=1, max_length=128)
    max_steps: int = Field(default=6, ge=1, le=12)


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    approved: bool


class AgentResponse(BaseModel):
    request_id: str | None
    thread_id: str
    status: str
    answer: str | None
    pending_action: dict | None
    model_calls: int


def result_payload(result):
    return AgentResponse(
        request_id=result.request_id,
        thread_id=result.thread_id,
        status=result.status,
        answer=result.answer,
        pending_action=result.pending_action,
        model_calls=result.model_calls,
    )


def sse(event, data):
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def create_app(service=None):
    """测试可注入假服务；生产环境在lifespan启动时创建真实服务。"""

    @asynccontextmanager
    async def lifespan(app):
        app.state.agent_service = service or build_default_service()
        try:
            yield
        finally:
            if service is None:
                app.state.agent_service.close()

    app = FastAPI(
        title="企业知识与工单Agent",
        version="1.0.0",
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
        return {"status": "ok", "service": "agent-api", "version": "1.0.0"}

    @app.post("/api/chat", response_model=AgentResponse)
    def chat(body: ChatRequest, agent_service=Depends(get_service)):
        try:
            result = agent_service.chat(body.question, body.thread_id, body.max_steps)
            return result_payload(result)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail="Agent暂时不可用") from error

    @app.post("/api/chat/stream")
    async def chat_stream(body: ChatRequest, agent_service=Depends(get_service)):
        async def events():
            yield sse("started", {"thread_id": body.thread_id})
            try:
                result = await asyncio.to_thread(
                    agent_service.chat,
                    body.question,
                    body.thread_id,
                    body.max_steps,
                )
                yield sse("result", result_payload(result).model_dump())
                yield sse("done", {"status": result.status})
            except Exception as error:
                yield sse(
                    "error",
                    {"type": type(error).__name__, "message": "Agent请求失败"},
                )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/approvals", response_model=AgentResponse)
    def approval(body: ApprovalRequest, agent_service=Depends(get_service)):
        try:
            return result_payload(agent_service.resume(body.thread_id, body.approved))
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail="该会话无法恢复") from error

    @app.get("/api/sessions/{thread_id}")
    def session(thread_id: str, agent_service=Depends(get_service)):
        try:
            return agent_service.get_session(thread_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="会话不存在") from error

    return app


app = create_app()
