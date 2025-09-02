from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    request_id: str
    query: str
    n: int = Field(default=1)
    top_p: float = Field(default=0.7)
    temperature: float = Field(default=1.0)
    max_tokens: int = Field(default=1024)
    seed: int = Field(default=42)

class GenerateResponse(BaseModel):
    response: str