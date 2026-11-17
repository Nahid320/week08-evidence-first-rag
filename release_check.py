"""Offline release checks.  No OpenAI calls or model downloads."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "streamlit_app.py", "app.py", "ui_backend.py", "llm_adapter.py",
    "reliability.py", "requirements.txt", "requirements-test.txt",
    ".gitignore", ".env.example", "README.md",
    ".streamlit/config.toml", ".github/workflows/tests.yml",
    "sample_pdfs/industrial_energy_report.pdf",
    "data/maintenance_incidents.json", "data/evaluation_questions.json",
]
REQUIRED_IGNORE_RULES = [".env", ".streamlit/secrets.toml", "chroma_data/", ".venv/", "outputs/"]


def check_files():
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    rules = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
    missing_rules = [rule for rule in REQUIRED_IGNORE_RULES if rule not in rules]
    if missing or missing_rules:
        raise RuntimeError(f"Missing files: {missing}; missing .gitignore rules: {missing_rules}")
    for file in ROOT.glob("*.py"):
        ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    for file in (ROOT / "tests").glob("*.py"):
        ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    print("Release files and Python syntax: OK", flush=True)


def run(cmd):
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    check_files()
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    run([sys.executable, "app.py", "safety-demo"])
    run([sys.executable, "app.py", "quality"])
    print("OFFLINE RELEASE CHECKS PASSED (not a deployment or live-model test).", flush=True)
