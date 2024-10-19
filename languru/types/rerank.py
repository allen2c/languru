from typing import List, Optional, Sequence, Text

from pydantic import BaseModel
from typing_extensions import Required, TypedDict


class RerankParams(TypedDict, total=False):
    query: Required[Text]
    documents: Required[Sequence[Text]]
    model: Optional[Text]
    top_k: Optional[int]
    truncation: bool


class RerankingResult(BaseModel):
    index: int
    document: Text
    relevance_score: float


class RerankingObject(BaseModel):
    results: List[RerankingResult]
    total_tokens: int
