import base64
import mimetypes
from pathlib import Path
from typing import Literal, Sequence

from dotenv import load_dotenv

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

DocType = Literal["pdf", "docx", "txt", "image"]
HistoryItem = dict  # {"role": "user" | "assistant", "content": str}


def _build_retriever_from_documents(documents: list[LCDocument]) -> BaseRetriever:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )


def _load_image_as_document(path: str) -> list[LCDocument]:
    """Extract text from an image using OpenAI's vision-capable model."""
    p = Path(path)
    mime, _ = mimetypes.guess_type(str(p))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"

    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Extract ALL readable text from this image exactly as it appears. "
                    "Preserve line breaks and order. If the image contains no text, "
                    "reply with an empty string."
                ),
            },
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }
    response = vision_llm.invoke([message])
    text = (
        response.content if isinstance(response.content, str) else str(response.content)
    ).strip()

    if not text:
        raise RuntimeError(
            "No text could be extracted from this image. "
            "Try a clearer scan or a different file."
        )
    return [LCDocument(page_content=text, metadata={"source": str(p)})]


def get_retriever(document: str, document_type: DocType) -> BaseRetriever:
    if document_type == "pdf":
        documents = PyPDFLoader(document).load()
    elif document_type == "docx":
        documents = Docx2txtLoader(document).load()
    elif document_type == "txt":
        documents = TextLoader(document, encoding="utf-8").load()
    elif document_type == "image":
        documents = _load_image_as_document(document)
    else:
        raise ValueError(f"Unsupported document_type: {document_type}")

    if not documents or all(not (d.page_content or "").strip() for d in documents):
        raise RuntimeError(
            "The document appears to be empty or unreadable. "
            "If it's a scanned PDF, try an image upload or a text-based PDF."
        )

    return _build_retriever_from_documents(documents)


def _to_messages(history: Sequence[HistoryItem] | None) -> list[BaseMessage]:
    msgs: list[BaseMessage] = []
    for h in history or []:
        role = (h.get("role") or "").lower()
        content = (h.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    # Keep the most recent turns to bound prompt size
    return msgs[-10:]


_CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question which may reference prior turns, "
            "reformulate the question as a standalone question that can be understood without the "
            "chat history. Do NOT answer the question. If no reformulation is needed, return it unchanged.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant answering questions about the user's document.\n"
            "Use the retrieved document context and the conversation history.\n"
            "Answer ONLY from the provided document context. If the context is insufficient, "
            "search through the document again and return the answer.\n\n"
            "Document context:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


def get_answer(
    question: str,
    retriever: BaseRetriever,
    history: Sequence[HistoryItem] | None = None,
) -> str:
    messages = _to_messages(history)

    # Reformulate the question using history so retrieval works on follow-ups.
    retrieval_query = question
    if messages:
        reformulator = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = _CONTEXTUALIZE_PROMPT | reformulator | StrOutputParser()
        try:
            reformulated = chain.invoke(
                {"history": messages, "question": question}
            ).strip()
            if reformulated:
                retrieval_query = reformulated
        except Exception:
            pass  # Fall back to the original question on any issue

    retrieved_docs = retriever.invoke(retrieval_query)
    if not isinstance(retrieved_docs, list):
        retrieved_docs = [retrieved_docs]
    context = "\n\n".join(d.page_content for d in retrieved_docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    qa_chain = _ANSWER_PROMPT | llm | StrOutputParser()
    return qa_chain.invoke(
        {"context": context, "history": messages, "question": question}
    )
