from __future__ import annotations

import json

from utils.models import SuggestionResponse


def _citation(title: str, doi: str, in_text: str = "Amin et al., 2025"):
    return {
        "id": "cite_1",
        "inText": in_text,
        "type": "Article",
        "articleType": "Journal",
        "title": title,
        "shortTitle": "",
        "abstract": "Abstract",
        "publication": "Journal of Climate Research",
        "year": 2025,
        "month": 1,
        "day": 1,
        "authors": [{"family": "Amin", "given": "Md Ruhul"}],
        "identifiers": {"doi": doi, "url": f"https://doi.org/{doi}"},
    }


def test_accept_suggestion_builds_references_and_switches_style(app_client):
    client, loaded = app_client
    store = loaded["utils.store"].store
    store.ensure_document("doc-1", "Climate Change", "APA")
    store.save_suggestion(
        {
            "suggestion_id": "sg_1",
            "document_id": "doc-1",
            "block_id": "p1",
            "paragraph_text": "Climate change affects ecosystems.",
            "title": "Climate Change",
            "heading": "Introduction",
            "cursor_context": "",
            "citation_style": "APA",
            "retry_of_suggestion_id": None,
            "dedupe_key": "dedupe-1",
            "status": "generated",
            "text": "Climate change affects ecosystems worldwide.",
            "response_json": {},
            "citations": [_citation("Climate Strategies", "10.1000/testdoi")],
        }
    )

    accept_response = client.post(
        "/suggestions/accept",
        json={
            "suggestionId": "sg_1",
            "documentId": "doc-1",
            "blockId": "p1",
            "acceptedText": "Climate change affects ecosystems worldwide.",
        },
    )
    assert accept_response.status_code == 200
    assert accept_response.json()["attachedCitationCount"] == 1

    references_response = client.get("/documents/doc-1/references")
    assert references_response.status_code == 200
    references = references_response.json()
    assert references["visible"] is True
    assert "Climate Strategies" in references["entries"][0]["text"]
    assert references["entries"][0]["label"] == "Amin et al., 2025"

    ieee_response = client.patch(
        "/documents/doc-1/citation-style",
        json={"citationStyle": "IEEE"},
    )
    assert ieee_response.status_code == 200
    ieee_payload = ieee_response.json()
    assert ieee_payload["entries"][0]["label"] == "[1]"


def test_websocket_supports_legacy_payload_shape(app_client, monkeypatch):
    client, loaded = app_client
    main_module = loaded["main"]

    async def fake_request_suggestion(_request):
        return SuggestionResponse(
            suggestionId="sg_ws",
            text="Generated sentence.",
            citations=[],
            status="generated",
            dedupeKey="abc123",
            latencyMs=10,
            debug={"path": "fake"},
        )

    monkeypatch.setattr(main_module, "request_suggestion", fake_request_suggestion)

    with client.websocket_connect("/suggest/doc-ws") as websocket:
        connection_message = websocket.receive_json()
        assert connection_message["status"] == "connected"
        websocket.send_text(
            json.dumps(
                {
                    "title": "Title",
                    "heading": "Intro",
                    "content": "Legacy payload content.",
                }
            )
        )
        result = websocket.receive_json()
        assert result["suggestionId"] == "sg_ws"
        assert result["text"] == "Generated sentence."


def test_chat_keeps_document_payload_separate(app_client):
    client, loaded = app_client
    store = loaded["utils.store"].store
    store.ensure_document("doc-chat", "Paper", "APA")
    store.upsert_block("doc-chat", "p1", "Original paragraph.")

    response = client.post(
        "/documents/doc-chat/chat/messages",
        json={
            "message": "Improve this paragraph",
            "action": "improve",
            "selectedText": "Original paragraph.",
            "context": {"document": True, "web": False, "library": False},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["assistantMessage"]
    assert payload["documentPayload"]
    assert payload["assistantMessage"] != payload["documentPayload"]


def test_edit_endpoint_returns_insert_below_payload(app_client):
    client, _ = app_client
    response = client.post(
        "/documents/doc-edit/edits",
        json={
            "selectionText": "This is a claim.",
            "instruction": "Convert to bullet list",
            "mode": "insert_below",
            "contextWindow": "This is a claim.",
            "targetBlockId": "p1",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["regenerateToken"]
    assert payload["insertBelowText"]


def test_bibtex_import_deduplicates(app_client):
    client, _ = app_client
    bibtex = """
    @article{amin2025,
      title={Climate Change and Global Impact},
      author={Amin, Md Ruhul},
      journal={Journal of Climate Research},
      year={2025},
      doi={10.1000/testdoi}
    }
    """

    first = client.post("/library/import/bibtex", json={"bibtex": bibtex, "documentId": "doc-lib"})
    assert first.status_code == 200
    assert first.json()[0]["duplicate"] is False

    second = client.post("/library/import/bibtex", json={"bibtex": bibtex, "documentId": "doc-lib"})
    assert second.status_code == 200
    assert second.json()[0]["duplicate"] is True


def test_doi_import_deduplicates(app_client, monkeypatch):
    client, loaded = app_client
    library_service = loaded["utils.library_service"]

    async def fake_enrich_single_citation(url: str, _cite_id: str):
        return _citation("Climate Strategies", "10.1000/testdoi", in_text="")

    async def fake_fetch_crossref_metadata(_doi: str):
        return {"abstract": "Metadata abstract"}

    monkeypatch.setattr(library_service, "enrich_single_citation", fake_enrich_single_citation)
    monkeypatch.setattr(library_service, "fetch_crossref_metadata", fake_fetch_crossref_metadata)

    first = client.post("/library/import/doi", json={"doi": "10.1000/testdoi", "documentId": "doc-doi"})
    assert first.status_code == 200
    assert first.json()["duplicate"] is False

    second = client.post("/library/import/doi", json={"doi": "10.1000/testdoi", "documentId": "doc-doi"})
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
