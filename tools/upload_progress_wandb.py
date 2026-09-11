#!/usr/bin/env python3
"""Upload PROGRESS.md to Weights & Biases as a versioned artifact + HTML panel.

Run this whenever PROGRESS.md is updated to keep a dated snapshot history in W&B.
"""

import argparse
import datetime
import os
import subprocess

import wandb


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROGRESS_PATH = os.path.join(REPO_ROOT, "PROGRESS.md")


def git_short_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def render_markdown_html(md_text: str) -> str:
    """Best-effort markdown -> HTML. Falls back to <pre> if no markdown lib."""
    try:
        import markdown  # type: ignore

        body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    except Exception:
        from html import escape

        body = f"<pre>{escape(md_text)}</pre>"
    return (
        "<html><head><meta charset='utf-8'>"
        "<style>body{font-family:-apple-system,Segoe UI,sans-serif;max-width:980px;"
        "margin:24px auto;padding:0 16px;line-height:1.5}"
        "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:4px 8px}"
        "code,pre{background:#f5f5f5;padding:2px 4px;border-radius:3px}"
        "pre{padding:8px;overflow-x:auto}</style></head>"
        f"<body>{body}</body></html>"
    )


def main():
    parser = argparse.ArgumentParser(description="Upload PROGRESS.md to W&B")
    parser.add_argument("--path", default=DEFAULT_PROGRESS_PATH, help="Path to PROGRESS.md")
    parser.add_argument("--project", default="bagel-spatial-reasoning", help="W&B project")
    parser.add_argument("--entity", default=None, help="W&B entity (team or username)")
    parser.add_argument("--artifact_name", default="progress-tracker", help="W&B artifact name")
    parser.add_argument("--run_name", default=None, help="Run name (default: progress-YYYY-MM-DD)")
    parser.add_argument("--note", default=None, help="Optional note saved in run config")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise SystemExit(f"PROGRESS.md not found: {args.path}")

    today = datetime.date.today().isoformat()
    run_name = args.run_name or f"progress-{today}"
    sha = git_short_sha()

    with open(args.path, "r") as f:
        md_text = f.read()

    print(f"[progress] Initializing W&B run: {run_name}")
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        job_type="progress-snapshot",
        tags=["progress-tracker", f"date:{today}"] + ([f"sha:{sha}"] if sha else []),
        config={
            "source_path": os.path.relpath(args.path, REPO_ROOT),
            "date": today,
            "git_sha": sha,
            "note": args.note,
            "num_chars": len(md_text),
            "num_lines": md_text.count("\n") + 1,
        },
        reinit=True,
    )

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type="document",
        description=f"PROGRESS.md snapshot ({today}, sha={sha or 'unknown'})",
        metadata={"date": today, "git_sha": sha},
    )
    artifact.add_file(args.path, name="PROGRESS.md")
    run.log_artifact(artifact)
    print(f"[progress] Logged artifact {args.artifact_name} ({len(md_text)} chars)")

    wandb.log({"progress_html": wandb.Html(render_markdown_html(md_text))})
    print("[progress] Logged rendered HTML panel")

    wandb.finish()
    print(f"[progress] Done: {run.url}")


if __name__ == "__main__":
    main()
