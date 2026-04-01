from typing import Optional

from pydantic import BaseModel, Field

class InitialState(BaseModel):
    user_id: Optional[str] = None
    subject: str
    subject_description: Optional[str] = None
    difficulty: Optional[str] = "auto"
    bloom_level: Optional[str] = None
    N: int = Field(..., ge=1, le=10)
    parallel_workflows: int = Field(default=3, ge=3, le=10)
    count: int = 0
