"""Week 8: final educational PDF RAG application.

Index a bundled example or up to three PDF uploads in a session-specific temporary DB.
The LLM is never called unless Generate answer is explicitly selected.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import streamlit as st
from dotenv import load_dotenv

from grounding import GroundedResponse
from hybrid_search import HybridSearcher, index_records
from reranker import load_cross_encoder
from ui_backend import (
    MAX_FILE_BYTES, MAX_FILES, MAX_QUESTION_CHARS, MAX_TOTAL_PAGES,
    answer, build_upload_records, retrieve, sample_records, upload_signature,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Evidence-first PDF RAG", page_icon="📚", layout="wide")
st.title("📚 Evidence-first PDF RAG")
st.caption("Week 8 · Hybrid search + Cross-Encoder reranking + checked citations")
st.info("Teaching demo: verifying that a quote exists does not prove a claim is true. Don't upload confidential files.")


def get_workspace() -> Path:
    if "workspace" not in st.session_state:
        st.session_state.workspace = tempfile.TemporaryDirectory(prefix="w8-rag-")
    return Path(st.session_state.workspace.name)


def get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key and key != "your_api_key_here":
        return key
    try:
        key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        key = ""
    return key if key != "your_api_key_here" else ""


def generate_answer(question: str, context: str) -> GroundedResponse | None:
    """Use a server-side API key, never put the key into prompts or the page."""
    from openai import OpenAI

    api_key = get_api_key()
    if not api_key:
        raise ValueError("API credentials not configured")
    from llm_adapter import request_grounded_answer
    client = OpenAI(api_key=api_key, timeout=25.0, max_retries=2)
    return request_grounded_answer(client, os.getenv("OPENAI_MODEL", "gpt-5-mini"), question, context)



with st.sidebar:
    st.header("1 · Documents")
    mode = st.radio("Data source", ["Bundled sample", "Upload PDFs"], index=0)
    uploads = []
    if mode == "Upload PDFs":
        files = st.file_uploader("PDF documents", type="pdf", accept_multiple_files=True,
                                 help=f"Up to {MAX_FILES} PDFs, {MAX_FILE_BYTES // (1024*1024)} MB each, {MAX_TOTAL_PAGES} pages total")
        if files:
            uploads = [(f.name, f.getvalue()) for f in files]
    signature = "bundled-sample-v1" if mode == "Bundled sample" else "uploads-" + upload_signature(uploads)
    build = st.button("Build / rebuild index", type="primary", use_container_width=True,
                      disabled=(mode == "Upload PDFs" and not uploads))

    st.header("2 · Retrieval settings")
    top_k = st.slider("Final passages", min_value=1, max_value=5, value=3)
    candidate_count = st.slider("Candidates before reranking", min_value=5, max_value=20, value=12)

if build:
    try:
        with st.spinner("Extracting, embedding and indexing PDFs..."):
            records = sample_records() if mode == "Bundled sample" else build_upload_records(uploads, get_workspace())
            count = index_records(records, db_path=get_workspace() / "chroma_data", overwrite=True)
            st.session_state.searcher = HybridSearcher(get_workspace() / "chroma_data")
            st.session_state.index_signature = signature
            st.session_state.index_count = count
            st.session_state.last_response = None
        st.success(f"Indexed {count} passages across the selected documents.")
    except Exception:
        st.error("Indexing failed. Check that your PDF contains selectable text and review the local terminal logs.")

indexed = (st.session_state.get("index_signature") == signature
           and "searcher" in st.session_state)
if not indexed:
    st.warning("Build the index for your selected documents before searching.")
else:
    st.success(f"Index ready · {st.session_state.get('index_count', '?')} passages")

st.subheader("Ask your documents")
with st.form("rag_form"):
    question = st.text_input("Question", value="Why did commissioning get postponed?",
                             max_chars=MAX_QUESTION_CHARS)
    action = st.radio("Mode", ["Retrieve only (no API call)", "Generate answer (OpenAI API)"],
                      horizontal=True)
    submitted = st.form_submit_button("Run", disabled=not indexed)

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    elif action.startswith("Generate") and not get_api_key():
        st.error("Missing API key. Configure OPENAI_API_KEY in your local .env or Streamlit Cloud secrets.")
    else:
        try:
            with st.spinner("Retrieving and reranking passages..."):
                if "reranker" not in st.session_state:
                    st.session_state.reranker = load_cross_encoder()
                results = retrieve(question, st.session_state.searcher, st.session_state.reranker,
                                   top_k=top_k, candidates=candidate_count)
            output = {"question": question, "action": action, **results, "answer": None}
            if action.startswith("Generate"):
                with st.spinner("Generating answer and checking quoted evidence..."):
                    output["answer"] = answer(question, results["hits"], generate_answer)
            st.session_state.last_response = output
        except Exception:
            st.session_state.last_response = None
            st.error("Search failed. Check model dependencies and the local terminal logs.")

last = st.session_state.get("last_response") if indexed else None
if last:
    st.divider()
    result = last.get("answer")
    if result is not None:
        status = result["status"]
        st.subheader("Answer")
        if status == "VERIFIED_QUOTES_ONLY":
            st.success("Citations and quotations validated (semantic truth not guaranteed).")
            for claim in result["claims"]:
                st.write(f"**{claim['statement']}** [{claim['source_id']}]")
                page = f"page {claim['page']}" if claim["page"] > 0 else "incident note"
                st.caption(f"{claim['filename']} · {page}")
                st.write(f"Evidence: “{claim['quote']}”")
        elif status in ("ABSTAINED", "ABSTAINED_NO_EVIDENCE", "NO_TRUSTED_EVIDENCE"):
            st.warning(f"{status}: No supported answer was returned.")
        elif status == "API_ERROR":
            st.error(result.get("message", "Model service unavailable."))
        elif status == "INVALID_MODEL_OUTPUT":
            st.error("INVALID_MODEL_OUTPUT: Model returned inconsistent answer/claim fields. Try again; if repeated, inspect the response schema and output-token limit.")
        elif status == "VERIFICATION_FAILED":
            st.error("VERIFICATION_FAILED: One or more claimed quotes or source IDs did not match retrieved evidence.")
        else:
            st.error(f"{status}: The response did not pass our checks.")
        if result.get("filtered_ids"):
            st.caption(f"Passages excluded by heuristic checks: {len(result['filtered_ids'])}")

    st.subheader("Retrieved evidence")
    if not last["hits"]:
        st.info("No passages found.")
    for rank, hit in enumerate(last["hits"], 1):
        metadata = hit["metadata"]
        page = f"page {metadata['page']}" if metadata["page"] > 0 else "incident note"
        with st.expander(f"{rank}. {metadata['source']} · {page} · {metadata['section']}"):
            st.text(hit["text"])
            st.caption(f"Cross-Encoder: {hit.get('cross_score', 0):.3f} · RRF: {hit.get('rrf_score', 0):.5f} (ranking scores, not confidence)")

st.divider()
st.caption("Educational prototype. Scanned PDFs require OCR. Temporary storage may persist until cleanup. "
           "A verified quotation does not prove factual correctness. Public deployments require access and cost controls.")
