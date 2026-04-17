from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class KnowledgeCreate(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None


class Question(BaseModel):
    question: str
    use_web: bool = Field(default=True, description="Allow web fallback")
    n_results: int = Field(default=5, ge=1, le=20, description="Top-K local docs")
