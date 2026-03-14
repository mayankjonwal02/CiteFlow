from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class CitationStyle(str, Enum):
    APA = "APA"
    MLA = "MLA"
    HARVARD = "Harvard"
    CHICAGO = "Chicago"
    IEEE = "IEEE"


class CitationAuthor(BaseModel):
    family: str = ""
    given: str = ""


class CitationIdentifiers(BaseModel):
    doi: str = ""
    url: str = ""


class CitationRecord(BaseModel):
    id: str = ""
    inText: str = ""
    type: str = ""
    articleType: str = ""
    title: str = ""
    shortTitle: str = ""
    abstract: str = ""
    publication: str = ""
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    authors: list[CitationAuthor] = Field(default_factory=list)
    identifiers: CitationIdentifiers = Field(default_factory=CitationIdentifiers)


class SuggestionRequest(BaseModel):
    documentId: str
    blockId: str = "paragraph-1"
    paragraphText: str = ""
    title: str = ""
    heading: str = ""
    cursorContext: str = ""
    citationStyle: CitationStyle = CitationStyle.APA
    retryOfSuggestionId: Optional[str] = None


class SuggestionResponse(BaseModel):
    suggestionId: str
    text: str
    citations: list[CitationRecord] = Field(default_factory=list)
    status: Literal["generated", "duplicate"]
    dedupeKey: str
    latencyMs: int = 0
    debug: dict[str, Any] = Field(default_factory=dict)


class SuggestionAcceptRequest(BaseModel):
    suggestionId: str
    documentId: str
    blockId: str
    acceptedText: str


class SuggestionAcceptResponse(BaseModel):
    suggestionId: str
    documentId: str
    blockId: str
    acceptedText: str
    attachedCitationCount: int
    referencesVisible: bool


class CitationStyleUpdateRequest(BaseModel):
    citationStyle: CitationStyle


class ReferenceEntry(BaseModel):
    citationKey: str
    label: str
    text: str
    style: CitationStyle
    sourceTitle: str = ""


class ReferencesResponse(BaseModel):
    documentId: str
    citationStyle: CitationStyle
    visible: bool
    editable: bool = False
    entries: list[ReferenceEntry] = Field(default_factory=list)


class DocumentManifestResponse(BaseModel):
    documentId: str
    citationStyle: CitationStyle
    blocks: dict[str, str] = Field(default_factory=dict)
    references: ReferencesResponse


class ChatContextToggles(BaseModel):
    document: bool = True
    web: bool = False
    library: bool = False


class ChatMessageRequest(BaseModel):
    message: str
    threadId: Optional[str] = None
    selectedText: str = ""
    action: str = "ask"
    context: ChatContextToggles = Field(default_factory=ChatContextToggles)


class ChatMessageResponse(BaseModel):
    threadId: str
    assistantMessage: str
    documentPayload: str = ""
    citations: list[CitationRecord] = Field(default_factory=list)
    contextSummary: dict[str, Any] = Field(default_factory=dict)


class EditRequest(BaseModel):
    selectionText: str
    instruction: str
    mode: str = "replace"
    contextWindow: str = ""
    targetBlockId: str = "paragraph-1"


class EditResponse(BaseModel):
    explanation: str
    replacementText: str = ""
    insertBelowText: str = ""
    regenerateToken: str
    citations: list[CitationRecord] = Field(default_factory=list)


class DOIImportRequest(BaseModel):
    doi: str
    documentId: Optional[str] = None


class BibtexImportRequest(BaseModel):
    bibtex: str
    documentId: Optional[str] = None


class LibrarySourceResponse(BaseModel):
    sourceId: str
    documentId: Optional[str] = None
    sourceType: str
    title: str
    citation: Optional[CitationRecord] = None
    duplicate: bool = False


class LibrarySearchRequest(BaseModel):
    query: str
    documentId: Optional[str] = None
    topK: int = 5


class LibrarySearchHit(BaseModel):
    sourceId: str
    title: str
    score: float
    snippet: str
    citation: Optional[CitationRecord] = None


class LibrarySearchResponse(BaseModel):
    query: str
    hits: list[LibrarySearchHit] = Field(default_factory=list)
