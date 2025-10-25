#!/usr/bin/env python3
import os
import json
import pathlib
import subprocess
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ---------- Paths & setup ----------
BASE = pathlib.Path(__file__).resolve().parent           # backend/
ROOT = BASE.parent                                       # project root
DASH = ROOT / "dashboard"                                # dashboard folder (must contain index.html)
DATA = BASE / "data"                                     # backend/data
DATA.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Product Opinions Backend", version="1.0.0")

# CORS (relax for local dev; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # e.g., ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static dashboard at /dashboard (NOT at "/")
app.mount("/dashboard", StaticFiles(directory=str(DASH), html=True), name="dashboard")


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def root():
    """Serve dashboard index at root."""
    index = DASH / "index.html"
    if not index.exists():
        return HTMLResponse(
            "<h2>Dashboard not found.</h2><p>Place your index.html under ./dashboard/</p>",
            status_code=404,
        )
    return FileResponse(str(index))


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/summaries")
def get_summaries():
    """
    Return the latest product opinions JSON if present.
    Path: backend/data/product_opinions.json
    Shape: either {"results":[...]} or an array [...]; frontend handles both.
    """
    out = DATA / "product_opinions.json"
    if not out.exists():
        return JSONResponse({"results": []})
    return FileResponse(str(out), media_type="application/json")


@app.post("/upload-input")
async def upload_input(file: UploadFile = File(...)):
    """
    Upload a dataset to backend/data/uploaded.ndjson (or .json).
    The /summarize endpoint will use this file by default.
    """
    suffix = ".ndjson" if file.filename.lower().endswith(".ndjson") else ".json"
    target = DATA / f"uploaded{suffix}"
    with open(target, "wb") as f:
        while chunk := await file.read(1 << 20):  # 1MB chunks
            f.write(chunk)
    return {"ok": True, "path": str(target.relative_to(ROOT))}


@app.post("/summarize")
def summarize(
    input_path: Optional[str] = Form(None),
    top: int = Form(20),
    min_mentions: int = Form(1),
    sample: int = Form(6),
    max_chars_each: int = Form(280),
    model: str = Form("mistral-large-latest"),
):
    """
    Run your summarize_products.py and return its JSON **in the response**.
    Also writes the same JSON to backend/data/product_opinions.json.
    """
    # Ensure API key is present for summarize_products.py (which calls Mistral)
    if not os.getenv("MISTRAL_API_KEY"):
        return JSONResponse({"error": "Missing MISTRAL_API_KEY"}, status_code=400)

    # Resolve input file:
    # 1) If input_path provided, use that; it can be absolute or relative to backend/
    # 2) Else prefer backend/data/uploaded.ndjson
    # 3) Else take the first *.ndjson or *.json found in backend/data/
    if input_path:
        inp = pathlib.Path(input_path)
        if not inp.is_absolute():
            inp = (BASE / input_path).resolve()
    else:
        uploaded_nd = DATA / "uploaded.ndjson"
        if uploaded_nd.exists():
            inp = uploaded_nd
        else:
            # fallback search
            candidates = list(DATA.glob("*.ndjson")) or list(DATA.glob("*.json"))
            if not candidates:
                return JSONResponse({"error": "No input dataset found in backend/data/"}, status_code=400)
            inp = candidates[0]

    if not inp.exists():
        return JSONResponse({"error": f"Input not found: {inp}"}, status_code=400)

    # Output path
    out_json = DATA / "product_opinions.json"

    # Path to the summarizer (should be in backend/ as summarize_products.py)
    summarizer = BASE / "summarize_products.py"
    if not summarizer.exists():
        return JSONResponse({"error": f"Missing {summarizer.name} in backend/"},
                            status_code=500)

    # Build command
    cmd = [
        "python", str(summarizer),
        "--in", str(inp),
        "--top", str(top),
        "--min-mentions", str(min_mentions),
        "--sample", str(sample),
        "--max-chars-each", str(max_chars_each),
        "--model", model,
        "--out-json", str(out_json),
    ]

    # Run it
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        # Return stderr to help debug
        return JSONResponse(
            {"error": "summarizer failed", "stderr": e.stderr[-2000:]},
            status_code=500,
        )

    # Read and return the freshly written JSON
    try:
        data = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as e:
        return JSONResponse({"error": f"Failed reading output JSON: {e}"}, status_code=500)

    return JSONResponse(data)
