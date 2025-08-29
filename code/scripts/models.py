# code/scripts/models.py
from __future__ import annotations
from typing import List, Optional, Annotated
from pydantic import BaseModel, Field, HttpUrl, constr, field_validator

Slug = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")]

class RawDocument(BaseModel):
    """
    Raw JSON as placed in data/raw_texts/*.json
    Only 'body' is required to be large; all other fields are concise metadata.
    """
    id: Slug
    title: str
    author: str
    orator: Optional[str] = None
    date: Annotated[str, Field(strip_whitespace=True)]  # ISO-like string (e.g., 1787-11-22)
    language: Annotated[str, Field(strip_whitespace=True, min_length=2)] = "en"
    public_domain: bool
    source_title: str
    source_url: Optional[HttpUrl] = None
    source_archive_url: Optional[HttpUrl] = None
    license: str
    tags: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    translator: Optional[str] = None
    editor: Optional[str] = None
    location: Optional[str] = None
    edition: Optional[str] = None
    notes: Optional[str] = None
    body: str

    @field_validator("orator", mode="before")
    
    @classmethod
    def default_orator(cls, v, info):
        return v or info.data.get("author")

class ProcessedChunk(BaseModel):
    parent_id: Slug
    chunk_id: Slug
    index: int = Field(..., ge=0)
    text: str
    approx_word_count: int
    est_duration_sec: float
    start_char: int
    end_char: int
    start_word: int
    end_word: int
    meta: dict

class ProcessedDocument(BaseModel):
    id: Slug
    title: str
    author: str
    orator: str
    date: str
    language: str
    public_domain: bool
    source_title: str
    source_url: Optional[HttpUrl] = None
    source_archive_url: Optional[HttpUrl] = None
    license: str
    tags: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    translator: Optional[str] = None
    editor: Optional[str] = None
    location: Optional[str] = None
    edition: Optional[str] = None
    notes: Optional[str] = None

    normalized: dict  # {"body": <cleaned_text>}
    word_count: int
    char_count: int
    est_read_time_sec: float
