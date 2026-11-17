# Evidence-first PDF RAG | Week 8 final portfolio project

A practical Retrieval-Augmented Generation (RAG) application with a Streamlit UI, PDF uploads, hybrid retrieval, Cross-Encoder reranking, cited answers, error handling, and offline evaluation.

**Status:** Offline test suite and deterministic demonstrations pass. Actual cloud deployment, Hugging Face model inference, and OpenAI generation must be tested on the intended machine/cloud account. The app is an **educational prototype**, not a production-hardened document service.

## Architecture

```text
PDF / incident records
  -> pypdf text extraction -> structure-aware chunks
  -> Sentence Transformer embeddings -> Chroma vector search
  -> BM25 keyword search -> RRF rank fusion
  -> Cross-Encoder reranker -> top passages
  -> optional OpenAI structured answer -> source ID + exact quote checks
  -> Streamlit answer / evidence / status
```

The two search methods and their fused rankings can be benchmarked with the CLI. A valid quotation proves the text is present, **not** that the claim logically follows from it. LLM-as-judge estimates also need independent human evaluation.

## Local run (Windows / PowerShell)

Extract this project as `AI-Projects/week08-rag-final`. Reuse your existing Week 5 environment:

```powershell
cd "$HOME\OneDrive\AI-Projects\week05-semantic-search"
.\.venv\Scripts\Activate.ps1
cd "..\week08-rag-final"
python -m pip install -r requirements.txt
python release_check.py
python -m streamlit run streamlit_app.py
```

Visit the local URL printed by Streamlit (`http://localhost:8501` by default).

1. Choose **Bundled sample**, then **Build / rebuild index**. The first run may download the embedding and reranking models from Hugging Face.
2. Ask `Why did commissioning get postponed?` using **Retrieve only (no API call)**; examine the cited page excerpts.
3. Copy `.env.example` to `.env`, supply `OPENAI_API_KEY`, and select **Generate answer (OpenAI API)**. The LLM request uses `max_output_tokens=2000` and distinguishes incomplete/unparsed responses. The API costs money.
4. Ask something absent from the documents, e.g. `What was the CEO's salary in 2025?`, and inspect whether it abstains. This behavior is tested, not guaranteed by instructions alone.
5. Optionally choose **Upload PDFs** to index up to 3 text-based PDFs, 5 MB each, with at most 50 pages total. Use **nonconfidential test PDFs only**.

To create a local `.env`:

```powershell
Copy-Item .env.example .env
```

Your `.env` should contain `OPENAI_API_KEY=...` and optionally `OPENAI_MODEL=gpt-5-mini`. **Never commit `.env` or `.streamlit/secrets.toml`.**

## Tests and benchmarks

```powershell
# Run all deterministic checks, examples, and Python unit tests
python release_check.py

# Or run individual commands
python -m unittest discover -s tests -v
python app.py safety-demo
python app.py quality

# Live evaluation: these download models / invoke APIs as indicated
python app.py index --overwrite
python app.py evaluate --top-k 3 --candidates 12
python app.py evaluate --top-k 3 --candidates 12 --with-llm --max-llm-cases 4
```

The first two evaluation commands involving `app.py index` and `app.py evaluate` use locally downloaded Hugging Face models. The final command with `--with-llm` also calls OpenAI and incurs charges. Reports go to `outputs/` and are excluded from Git.

**Evaluation details:** 12 answerable and 2 unanswerable labeled queries. Retrieval Precision@k, Recall@k, Hit@k, and MRR@k are computed only over the 12 answerable queries; the two unanswerable cases are included in the optional answer/abstention tests. An LLM judge estimates faithfulness on labeled example answers; it does not provide ground truth.

`requirements-test.txt` contains lightweight dependencies for offline tests. GitHub Actions runs `python -m unittest discover -s tests -v` and the deterministic demos on every push and pull request, without Hugging Face downloads or API calls.

## Cloud deployment

Follow [DEPLOYMENT.md](DEPLOYMENT.md). Community Cloud requires a GitHub repository, a `requirements.txt`, and the entrypoint `streamlit_app.py`. Store `OPENAI_API_KEY` in **Streamlit app secrets**, never in your Git history. For a public portfolio demo, **do not add a paid API key** until you implement authentication, user quotas, and spending controls. Retrieval-only mode still works without the key.

Model sizes and dependency downloads may exceed Community Cloud memory or startup limits. **Successful local operation does not establish deployability on Community Cloud.** If deployment fails with a resource limit, optimize/caching or choose a more capable host; see official Streamlit troubleshooting.

## Project files

- `streamlit_app.py` – user interface (bundle/upload, retrieval, answers, evidence)
- `ui_backend.py` – upload validation, chunk processing and retrieval adapter
- `llm_adapter.py` – OpenAI structured response adapter with incomplete/unparsed diagnostics, 2,000 output-token cap
- `hybrid_search.py`, `reranker.py` – hybrid retrieval and Cross-Encoder
- `reliability.py`, `grounding.py` – limited safety controls and quote/source checks
- `chunkers.py`, `corpus.py` – PDF chunking and labeled sample documents
- `evaluation.py`, `answer_quality.py` – retrieval and generation quality benchmarks
- `app.py` – CLI experiments, evaluation and safety demos
- `release_check.py` – offline release verification
- `tests/` – deterministic tests; `data/`, `sample_pdfs/` – fictional demonstrations
- `.github/workflows/tests.yml` – CI; `requirements-test.txt` – lightweight test dependencies

## Limitations and risks

- PDF extraction may fail on scans, multi-column layouts and complicated tables. OCR is not implemented.
- Chunking, relevance ranking and factual support are imperfect; *verification* refers only to quotation/source-ID checks.
- The pattern-based prompt-injection filter detects a few obvious attacks, not arbitrary malicious instructions.
- Uploaded content is temporarily persisted as extracted text in a per-session Chroma database; it may survive until cleanup. This is not secure multi-tenant isolation.
- A public unauthenticated app with a configured OpenAI key can generate unexpected charges. Do not upload sensitive documents or enable paid generation publicly without access controls and budget enforcement.
- No permanent shared user store, guarantees about deletion, malware scanning, observability, or full deployment testing are provided.

References: [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management), [Streamlit resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app), [OpenAI Python structured outputs](https://github.com/openai/openai-python/blob/main/examples/responses/structured_outputs.py).
