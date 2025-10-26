# api/index.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import datetime

# Print on import so you can see in Vercel logs that the module loaded
print(">>> Importing api.index at", datetime.utcnow().isoformat(), "UTC")

app = FastAPI(title="Unveil", version="HACKOH/IO")

# super-safe minimal home: proves routing works
@app.get("/api/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

# If your PAGE_INDEX template isn't available yet, this still works
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

# Local dev runner only (Vercel ignores this block)
if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run("api.index:app", host="0.0.0.0", port=int(os.getenv("PORT","8080")), reload=True)
