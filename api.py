'''
 * @Author       : MatthewZhang
 * @Date         : 2026-04-04 10:53:38
 * @Description  : 
'''
import time

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import configure_settings, load_config
from service import RagService

from logging_config import setup_logging
setup_logging()


class QueryRequest(BaseModel):
    question: str


class SourceNode(BaseModel):
    score: float
    source_file: str
    heading: str
    position: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceNode]
    time_seconds: float


config = load_config()
configure_settings(config)
rag_service = RagService(config)

app = FastAPI(title="RAG QA API")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    start = time.time()
    result = rag_service.query(request.question)
    elapsed = round(time.time() - start, 2)

    return QueryResponse(
        answer=str(result["answer"]),
        sources=[SourceNode(**source) for source in result["sources"]],
        time_seconds=elapsed,
    )

@app.post("/query_stream")
def query_stream(request: QueryRequest):
    return StreamingResponse(
        rag_service.query_stream(request.question),
        media_type="text/plain",
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    content = file.file.read().decode("utf-8")
    return rag_service.upload_text(file.filename, content)
