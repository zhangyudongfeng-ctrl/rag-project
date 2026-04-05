import time

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from config import configure_settings, load_config
from service import RagService


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    content = file.file.read().decode("utf-8")
    return rag_service.upload_text(file.filename, content)
