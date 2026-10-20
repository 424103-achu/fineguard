from pydantic import BaseModel, Field, HttpUrl

DISCLAIMER = ("This tool highlights clauses that may be unfair to consumers based on a research "
              "dataset annotated under EU consumer law. It is not legal advice and may miss or "
              "misclassify clauses.")


class TextRequest(BaseModel):
    text: str = Field(min_length=20, max_length=500_000)
    service_name: str | None = Field(default=None, max_length=200)


class UrlRequest(BaseModel):
    url: HttpUrl
    service_name: str | None = Field(default=None, max_length=200)


class FlaggedClause(BaseModel):
    index: int                       # position of the sentence in the document
    sentence: str
    categories: list[str]
    confidence: dict[str, float]     # probability for each flagged category


class CategorySummary(BaseModel):
    category: str
    count: int
    weight: float                    # normalised weight of this category in the score


class AnalysisResponse(BaseModel):
    source: str
    service_name: str | None
    n_sentences: int
    n_flagged: int
    risk_score: float
    grade: str
    categories: list[CategorySummary]
    flagged: list[FlaggedClause]
    sentences: list[str]             # every sentence in order, for the document view
    processing_ms: float
    model_version: str
    weights_version: str
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    weights_version: str | None = None
    labels: list[str] = []
