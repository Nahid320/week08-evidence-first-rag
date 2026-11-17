# Week 8 final: GitHub and deployment walkthrough

## A. Verify and commit locally

In VS Code PowerShell, from `week08-rag-final`:

```powershell
python release_check.py
# Confirm that secrets or local databases are not staged:
git init
git branch -M main
git add .
git status --short
```

If `.env`, `.streamlit/secrets.toml`, `.venv/`, `chroma_data/`, or `outputs/` appear as staged files, **do not commit**. Check `.gitignore`, and unstage them (`git rm --cached <path>` for already tracked files). Repository files intentionally include example text, demonstration PDFs, tests, configuration and code.

Create an **empty public** GitHub repository named `week08-evidence-first-rag` at https://github.com/new. Do not ask GitHub to initialize it with a README. Then:

```powershell
git commit -m "Complete Week 8 evaluated PDF RAG app"
git remote add origin https://github.com/YOUR-USERNAME/week08-evidence-first-rag.git
git push -u origin main
```

Replace `YOUR-USERNAME`. If the remote already exists, use `git remote -v` to inspect it instead of adding it again. Visit the repository: confirm your README and GitHub Actions job, and check `.env` was not uploaded.

## B. Deploy to Streamlit Community Cloud

1. Open https://share.streamlit.io/ and connect your GitHub account.
2. Click **Create app** / **New app** and select repository `week08-evidence-first-rag`, branch `main`, entrypoint `streamlit_app.py`.
3. In **Advanced settings**, choose **Python 3.12** (supported and widely compatible with the ML dependencies). Use the same Python version for local reproduction when possible.
4. For the safest *public* demo, leave `OPENAI_API_KEY` unset. Retrieval-only mode still works. Configuring your private paid key on an unauthenticated public demo risks abuse and charges. If you deploy privately with adequate access controls, set secrets in Streamlit's Advanced settings, using TOML such as:

   ```toml
   OPENAI_API_KEY = "your_secret_here"
   ```

   Do not place real secret values in `secrets.toml` tracked by Git.
5. Deploy. Check logs for dependency/model download errors, memory limits, and timeouts. Build the sample index, use **Retrieve only**, and (only if authorized and cost-controlled) test **Generate answer**.

**This project has not been deployed by the assistant.** Streamlit Community Cloud has resource limits that may be insufficient for Chroma + Sentence Transformers + Cross-Encoder, especially with multiple users. If it hits resource limits, keep it as a GitHub portfolio project or deploy on a host with sufficient RAM. No claim of reliable public deployment is made.

## C. Final acceptance checklist

- [ ] `python release_check.py` passes locally.
- [ ] Streamlit loads the bundled sample and can build its index.
- [ ] Retrieve-only results include relevant excerpts and page references.
- [ ] Generate-answer mode produces a cited answer for the commissioning question.
- [ ] An unsupported question is tested for abstention, without assuming it will always abstain.
- [ ] Your GitHub repository includes files, tests, CI and README, but no secrets or local databases.
- [ ] If deployed, test the live URL and confirm costs and access restrictions are acceptable.

Official guides: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy and https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
