import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_functions import get_answer, get_retriever

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_ROOT = Path(__file__).resolve().parent / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

# document_id -> LangChain retriever for /generate_answer
retrievers = {}


class ChatHistory(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=datetime.now)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class GenerateAnswerBody(BaseModel):
    question: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    history: List[HistoryMessage] = Field(default_factory=list)


chat_history: List[ChatHistory] = []


def infer_document_type(filename: str) -> Literal["pdf", "docx", "txt", "image"] | None:
    ext = Path(filename or "").suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in (".docx", ".doc"):
        return "docx"
    if ext == ".txt":
        return "txt"
    if ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif"):
        return "image"
    return None


def count_pdf_pages(path: Path) -> int | None:
    try:
        return len(PyPDFLoader(str(path)).load())
    except Exception:
        return None


@app.get("/chat_history")
def get_chat_history():
    return {"message": "Chat history fetched successfully", "data": chat_history}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_id: str | None = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    doc_type = infer_document_type(file.filename)
    if doc_type is None:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use PDF, Word, TXT, or common image formats.",
        )

    try:
        did = str(uuid.UUID(document_id)) if document_id else str(uuid.uuid4())
    except ValueError:
        did = str(uuid.uuid4())

    safe_suffix = Path(file.filename).suffix
    dest = UPLOAD_ROOT / f"{did}{safe_suffix}"

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    dest.write_bytes(raw)

    try:
        r = get_retriever(str(dest), doc_type)
        retrievers[did] = r
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e)) from e

    pages: int | None = None
    if doc_type == "pdf":
        pages = count_pdf_pages(dest)

    return {
        "message": "Document added successfully",
        "document_id": did,
        "document_type": doc_type,
        "pages": pages,
    }


@app.post("/generate_answer")
def generate_answer(body: GenerateAnswerBody):
    r = retrievers.get(body.document_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Document not found. Upload the document first.")

    chat_history.append(ChatHistory(role="user", content=body.question))
    try:
        answer = get_answer(
            body.question,
            r,
            history=[h.model_dump() for h in body.history],
        )
    except Exception as e:
        chat_history.pop()
        raise HTTPException(status_code=500, detail=str(e)) from e

    answer_text = answer if isinstance(answer, str) else str(answer)
    chat_history.append(ChatHistory(role="assistant", content=answer_text))
    return {
        "message": "Answer generated successfully",
        "data": answer_text,
        "chat_history": chat_history,
    }
