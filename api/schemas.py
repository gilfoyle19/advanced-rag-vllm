# api/schemas.py
from pydantic import BaseModel, field_validator
from typing import List, Optional


class ChatRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty.")
        if len(v) > 2000:
            raise ValueError("Question too long. Max 2000 characters.")
        return v.strip()


class DocumentItem(BaseModel):
    page_content: str
    source: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    documents: List[DocumentItem] = []
    web_search_used: bool = False
