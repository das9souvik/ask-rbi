"""Deploy the app and the prebuilt index to a Hugging Face Space.

Usage (needs HF_TOKEN with write access in your environment or .env):
    python scripts/deploy_space.py --space <your-hf-username>/ask-rbi

Only the files the app needs are uploaded: app.py, the ask_rbi package, the index and
requirements. RBI PDFs are not uploaded; answers link to RBI's own copies.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "space_build"


def stage() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()
    shutil.copy(ROOT / "app.py", STAGE / "app.py")
    shutil.copy(ROOT / "requirements.txt", STAGE / "requirements.txt")
    shutil.copy(ROOT / "space" / "README.md", STAGE / "README.md")
    shutil.copytree(ROOT / "ask_rbi", STAGE / "ask_rbi", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "index", STAGE / "index")
    (STAGE / "data").mkdir()
    shutil.copy(ROOT / "data" / "manifest.toml", STAGE / "data" / "manifest.toml")
    if not (STAGE / "index" / "chunks.jsonl").exists():
        raise SystemExit("index/ is empty. Run `python -m ask_rbi.cli build` first.")
    return STAGE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="e.g. souvikdas09/ask-rbi")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (a Hugging Face token with write access).")
    # The running Space only needs to call Inference Providers. Give it a separate,
    # fine-grained token limited to that, so a leak can't touch your repos.
    groq_key = os.environ.get("GROQ_API_KEY")
    space_token = os.environ.get("HF_SPACE_TOKEN")
    if not space_token and not groq_key:
        raise SystemExit(
            "Set GROQ_API_KEY (free tier) or HF_SPACE_TOKEN: a fine-grained Hugging Face token with only "
            "'Make calls to Inference Providers' permission. It is stored as a private Space secret."
        )
    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    # Secrets are private to the Space; values are never printed.
    if groq_key:
        api.add_space_secret(args.space, "GROQ_API_KEY", groq_key)
    if space_token:
        api.add_space_secret(args.space, "HF_TOKEN", space_token)
    folder = stage()
    api.upload_folder(folder_path=str(folder), repo_id=args.space, repo_type="space",
                      commit_message="Deploy Ask RBI")
    print(f"Deployed: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
