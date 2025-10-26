from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from datetime import datetime

app = FastAPI(title="Unveil", version="HACKOH/IO")

@app.get("/api/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(
        f"""
        <!doctype html><meta charset="utf-8">
        <title>Unveil</title>
        <body style="font:16px system-ui;padding:32px">
          <h1>Unveil is running ✅</h1>
          <p>UTC: {datetime.utcnow().isoformat()}</p>
          <p><a href="/api/health">/api/health</a></p>
        </body>"""
    )

# Vercel handler
handler = app
