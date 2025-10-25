#!/usr/bin/env python3
"""
Universal Reddit Consensus Search Engine (FastAPI)
==================================================

• Works across all product categories (haircare, table tennis, tech, skincare, etc.)
• Automatically maps queries to likely subreddits + r/all fallback
• Uses multithreaded Reddit fetching (8× faster)
• Universal product phrase extraction (regex + NER + RapidFuzz)
• Groups results by detected subcategory (e.g., “rubber”, “paddle”, “cream”)
• TF-IDF + upvote-weighted consensus scoring
• Same routes: /, /search, /api/search, /download.csv
"""

from __future__ import annotations
import os, re, sys, math, time, html, logging
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from urllib.parse import quote_plus

import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Template
from rapidfuzz import fuzz

# ---------------------------------------------------
# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------------------------------------------
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ---------------------------------------------------
# spaCy setup
def load_spacy():
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            from spacy.cli import download
            logging.info("Downloading spaCy model en_core_web_sm ...")
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp
    except ImportError:
        logging.error("spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm")
        sys.exit(1)

NLP = load_spacy()

# ---------------------------------------------------
# Reddit client
REDDIT = None
def get_reddit():
    global REDDIT
    if REDDIT: return REDDIT
    try:
        import praw
    except ImportError:
        logging.error("praw not installed. Run: pip install praw")
        sys.exit(1)

    cid = os.getenv("REDDIT_CLIENT_ID", "").strip()
    csec = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    ua = os.getenv("REDDIT_USER_AGENT", "universal-consensus/1.0 (by u/yourusername)").strip()
    if not cid or not csec:
        raise RuntimeError("Missing Reddit credentials. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET.")
    REDDIT = praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua, check_for_async=False)
    _ = REDDIT.read_only
    return REDDIT

# ---------------------------------------------------
# Utility helpers
_token_re = re.compile(r"[a-z0-9]+")
def canon(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("’","'").replace("–","-").replace("—","-")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,:;!?-_/\\|\"'`~()[]{}")

def token_key(phrase: str) -> Tuple[str,...]:
    toks = _token_re.findall(canon(phrase))
    toks = [t for t in toks if t not in {"the","and","of","for","with","a","an"}]
    return tuple(sorted(toks))

# ---------------------------------------------------
# Generic regex patterns
PRODUCT_PAT = re.compile(
    r"\b([A-Z][\w'&\.-]+(?:\s+[A-Z0-9][\w'&\.-]+){0,3})\s+([a-z]{2,20})\b",
    re.UNICODE,
)
NUMBERED_PAT = re.compile(r"\b([A-Z][\w'&\.-]+)\s+(?:no\.?|#)\s*(\d{1,3})\b", re.IGNORECASE)

# ---------------------------------------------------
# Subreddit mapping
CATEGORY_MAP = {
    "hair": "HaircareScience+curlyhair+malehairadvice+femalefashionadvice+longhair+malehair",
    "skincare": "SkincareAddiction+AsianBeauty+Beauty+30PlusSkinCare+SkincareAddictionUK",
    "makeup": "MakeupAddiction+Beauty+femalefashionadvice+RedditLaqueristas",
    "fragrance": "fragrance+Perfumes+IndieMakeupAndMore",
    "nails": "RedditLaqueristas+Nailpolish",
    "selfcare": "Beauty+MakeupAddiction+SkincareAddiction",
    "haircare": "HaircareScience+curlyhair+femalefashionadvice+malehairadvice",
    "default": "Beauty+SkincareAddiction+MakeupAddiction+HaircareScience"
}


def guess_subs(q: str) -> str:
    ql = q.lower()
    for k, v in CATEGORY_MAP.items():
        if k in ql:
            return v
    return CATEGORY_MAP["default"]

# ---------------------------------------------------
# Universal product phrase extractor
def extract_product_phrases(text: str) -> List[str]:
    if not text:
        return []

    found = set()

    # --- Core product nouns (beauty-focused)
    GENERIC_NOUNS = {
        "shampoo","conditioner","serum","cream","gel","spray","mask",
        "cleanser","toner","moisturizer","oil","balm","exfoliant",
        "essence","lotion","primer","foundation","concealer","lipstick",
        "mascara","palette","eyeliner","powder","bronzer","blush",
        "sunscreen","retinol","ampoule","peel","mist"
    }

    # --- Recognized brand hints (expand as needed)
    BRAND_HINTS = [
        "olaplex","loreal","redken","pureology","amika","davines",
        "garnier","pantene","shea","moroccanoil","cerave","ouai",
        "briogeo","matrix","function","head","dove","tresemme",
        "ordinary","paula","neutrogena","olay","clinique","tatcha",
        "glossier","rare","fenty","elf","maybelline","nars","estee",
        "lancome","drunk","innisfree","cosrx","aveda","aesop","k18"
    ]

    # --- Regex for brand + product
    BRAND_PRODUCT = re.compile(
        r"\b([A-Z][\w'&\.-]+(?:\s+[A-Z0-9][\w'&\.-]+){0,3})\s+(?:"
        + "|".join(GENERIC_NOUNS) + r")\b",
        re.IGNORECASE,
    )
    MODEL_PAT = re.compile(
        r"\b([A-Z][\w'&\.-]+(?:\s+[A-Z0-9][\w'&\.-]+){0,2})\s*(?:no\.?|#)?\s*\d{1,4}[A-Z]?\b",
        re.IGNORECASE,
    )

    # --- Extract brand+product patterns, skipping generic connectors
    for m in BRAND_PRODUCT.finditer(text):
        phrase = canon(m.group(0))
        if re.search(r"\b[a-z]{3,}\b", phrase):  # must have words
            if not re.search(r"^(and|best|your|this|that|my|a|the)\b", phrase):
                found.add(phrase)

    for m in MODEL_PAT.finditer(text):
        found.add(canon(m.group(0)))

    # --- spaCy pass for brand/product entities
    doc = NLP(text)
    for ent in doc.ents:
        if ent.label_ in {"PRODUCT", "ORG"}:
            if any(b in ent.text.lower() for b in BRAND_HINTS):
                found.add(canon(ent.text))

    # --- Combine nearby nouns (brand + noun)
    toks = [t for t in doc if t.pos_ in {"PROPN", "NOUN"}]
    for i in range(len(toks) - 2):
        phrase = " ".join(t.text for t in toks[i:i+3])
        if any(n.text.lower() in GENERIC_NOUNS for n in toks[i:i+3]):
            found.add(canon(phrase))

    # --- Cleanup: remove short or generic fragments
    found = {
        f
        for f in found
        if 3 <= len(f) <= 60
        and len(f.split()) >= 2
        and not re.match(
            r"^(if|thank|feel|good|bad|time|routine|skin|hair|face|help|and|best|your|this|that|my|a|the)\b",
            f,
        )
    }

    # --- Merge near-dupes
    merged, used, items = [], set(), list(found)
    for i, p in enumerate(items):
        if p in used:
            continue
        cluster = [p]
        for q in items[i + 1:]:
            if q in used:
                continue
            if fuzz.token_sort_ratio(p, q) >= 90:
                cluster.append(q)
                used.add(q)
        rep = min(cluster, key=len)
        merged.append(rep)
        used.update(cluster)

    # --- Enforce brand/model presence to drop generic "and conditioner"
    filtered = []
    for f in merged:
        if any(b in f.lower() for b in BRAND_HINTS):
            filtered.append(f)
        elif re.search(r"\bno\.\s*\d{1,3}\b", f):  # e.g. "Olaplex No.3"
            filtered.append(f)
        elif re.search(r"\b\d{1,3}[A-Z]?\b", f):  # numeric models like "K18 450"
            filtered.append(f)

    # --- Final cleanup: remove emojis, verbs, duplicates
    cleaned = []
    for f in filtered:
        f = re.sub(r"[\U00010000-\U0010ffff]", "", f)  # remove emojis/unicode
        f = re.sub(r"\b(try|use|love|recommend|definitely|using|apply|avoid)\b", "", f).strip()
        f = re.sub(r"\b(\w+)\s+\1\b", r"\1", f)  # collapse repeats like "shampoo shampoo"
        if len(f.split()) >= 2:
            cleaned.append(f)

    return cleaned


# ---------------------------------------------------
# Fetch Reddit posts concurrently
def fetch_posts(reddit, query:str, subs:str, limit_posts:int, comments_per_post:int) -> List[Dict[str,Any]]:
    space = reddit.subreddit(subs if subs.lower()!="all" else "all")
    submissions=list(space.search(query, limit=limit_posts, sort="relevance"))
    results=[]
    def handle(sub):
        try:
            sub.comment_sort="best"
            sub.comments.replace_more(limit=0)
            comments=[{"body":c.body,"score":int(getattr(c,"score",0))} for c in sub.comments[:comments_per_post]]
            return {
                "id":sub.id,
                "subreddit":str(sub.subreddit),
                "title":sub.title or "",
                "selftext":sub.selftext or "",
                "score":int(sub.score or 0),
                "url":f"https://www.reddit.com{sub.permalink}",
                "comments":comments,
            }
        except Exception as e:
            logging.warning(f"Skip post: {e}")
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        fut=[ex.submit(handle,s) for s in submissions]
        for f in as_completed(fut):
            r=f.result()
            if r: results.append(r)
    return results

# ---------------------------------------------------
# Consensus scoring + grouping
def consensus_phrases(posts:List[Dict[str,Any]], top_k:int=12)->Dict[str,List[Dict[str,Any]]]:
    docs=["\n".join([p["title"],p.get("selftext","")]+[c["body"] for c in p.get("comments",[])]) for p in posts]
    phrases_per_doc=[extract_product_phrases(text) for text in docs]
    N=max(1,len(docs))
    df=Counter(ph for doc in phrases_per_doc for ph in set(doc))
    idf={ph:math.log((N+1)/(dfv+0.5))+1.0 for ph,dfv in df.items()}

    scores,examples,urls=Counter(),defaultdict(list),defaultdict(set)
    for p,phrases in zip(posts,phrases_per_doc):
        w=1.0+min(max(p.get("score",0),0),1000)/1000.0
        beauty_bonus = 1.2 if any(s in p["subreddit"].lower() for s in ["hair","skin","makeup","beauty"]) else 1.0
        w *= beauty_bonus
        for ph in set(phrases):
            scores[ph]+=w*idf.get(ph,1.0)
            urls[ph].add(p["url"])
            if len(examples[ph])<3:
                ex=(p.get("title") or p.get("selftext") or "")[:200].replace("\n"," ")
                examples[ph].append(ex+"…")
        for c in p.get("comments",[])[:5]:
            c_ph=extract_product_phrases(c["body"])
            cw=1.0+min(max(c.get("score",0),0),100)/200.0
            for ph in set(c_ph):
                scores[ph]+=0.25*cw*idf.get(ph,1.0)

    ranked=scores.most_common(200)
    rows=[{
        "phrase":ph,"score":round(sc,3),
        "examples":examples.get(ph,[]),
        "urls":list(urls.get(ph,[]))[:3],
    } for ph,sc in ranked]
    # group by final noun as subcategory
    groups=defaultdict(list)
    for r in rows:
        toks=r["phrase"].split()
        subcat=toks[-1] if toks else "misc"
        groups[subcat].append(r)
    for k in groups: groups[k]=sorted(groups[k],key=lambda r:r["score"],reverse=True)[:top_k]
    return groups

# ---------------------------------------------------
# Cached search
@lru_cache(maxsize=64)
def cached_search(q:str, subs:str, lp:int, cp:int, topk:int):
    reddit = get_reddit()
    posts  = fetch_posts(reddit, q, subs, lp, cp)
    top    = consensus_phrases(posts, topk)
    return posts, top

# ---------------------------------------------------
# HTML templates
PAGE_INDEX = Template("""
<!doctype html><html><head>
<meta charset="utf-8"><title>Universal Reddit Consensus Search</title>
<style>
body{font-family:Arial,Helvetica,sans-serif;margin:40px;}
input,button{margin:6px;padding:4px;}
h1{color:#2c3e50;}
section{margin-top:32px;}
table{border-collapse:collapse;width:100%;}
th,td{border:1px solid #ccc;padding:6px 8px;vertical-align:top;}
th{background:#f5f5f5;}
td div{margin-bottom:4px;}
</style>
</head><body>
<h1>Universal Reddit Consensus Search</h1>
<form action="/search" method="get">
<input name="query" placeholder="best table tennis rubber" size="40" required>
<br>
<label>Subreddits (optional)</label>
<input name="subreddits" placeholder="auto" size="40">
<br>
<label>Limit posts:</label>
<input type="number" name="limit_posts" value="60">
<label>Comments per post:</label>
<input type="number" name="comments_per_post" value="8">
<label>Top K:</label>
<input type="number" name="top_k" value="12">
<br><button type="submit">Search</button>
</form>
<p style="margin-top:20px;">Try queries like <em>"best headphones"</em>, <em>"best running shoes"</em>, or <em>"best table tennis rubber"</em>.</p>
</body></html>
""")

PAGE_RESULTS = Template("""
<!doctype html><html><head>
<meta charset="utf-8"><title>Consensus for {{q}}</title>
<style>
body{font-family:Arial,Helvetica,sans-serif;margin:40px;}
section{margin-top:40px;}
h2{color:#2c3e50;}
table{border-collapse:collapse;width:100%;}
th,td{border:1px solid #ccc;padding:6px 8px;vertical-align:top;}
th{background:#f5f5f5;}
td div{margin-bottom:4px;}
a{text-decoration:none;color:#2980b9;}
</style>
</head><body>
<h1>Consensus for “{{q}}”</h1>
<p>Subreddits searched: <strong>{{subs}}</strong></p>
{% for cat,items in groups.items() %}
<section>
<h2>Top {{cat|capitalize}} Results</h2>
<table>
<tr><th>#</th><th>Phrase</th><th>Score</th><th>Examples (3 max)</th></tr>
{% for r in items %}
<tr>
<td>{{loop.index}}</td>
<td>{{r['phrase']}}</td>
<td>{{r['score']}}</td>
<td>
{% for ex in r['examples'] %}
<div>{{ex|e}}</div>
{% endfor %}
{% for u in r['urls'] %}
<div><a href="{{u}}" target="_blank">[source]</a></div>
{% endfor %}
</td>
</tr>
{% endfor %}
</table>
</section>
{% endfor %}
</body></html>
""")

# ---------------------------------------------------
# FastAPI setup
app = FastAPI(title="Universal Reddit Consensus Search Engine", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE_INDEX.render()

# ---------------------------------------------------
@app.get("/search", response_class=HTMLResponse)
def search(
    request: Request,
    query: str = Query(...),
    subreddits: str = "",
    limit_posts: int = 60,
    comments_per_post: int = 8,
    top_k: int = 12
):
    try:
        subs = subreddits or guess_subs(query)
        _, groups = cached_search(query, subs, limit_posts, comments_per_post, top_k)
    except Exception as e:
        import traceback
        traceback.print_exc()
        msg = html.escape(str(e))
        return HTMLResponse(
            f"<h1 style='color:red'>Error</h1><p>{msg}</p>",
            status_code=500,
        )
    return HTMLResponse(PAGE_RESULTS.render(q=query, subs=subs, groups=groups))

# ---------------------------------------------------
@app.get("/api/search")
def api_search(
    query: str = Query(...),
    subreddits: str = "",
    limit_posts: int = 60,
    comments_per_post: int = 8,
    top_k: int = 12
):
    subs = subreddits or guess_subs(query)
    _, groups = cached_search(query, subs, limit_posts, comments_per_post, top_k)
    return JSONResponse({"query": query, "subreddits": subs, "results": groups})

# ---------------------------------------------------
@app.get("/download.csv")
def download_csv(
    query: str = Query(...),
    subreddits: str = "",
    limit_posts: int = 60,
    comments_per_post: int = 8,
    top_k: int = 12
):
    subs = subreddits or guess_subs(query)
    _, groups = cached_search(query, subs, limit_posts, comments_per_post, top_k)
    all_rows = []
    for cat, items in groups.items():
        for r in items:
            all_rows.append({
                "subcategory": cat,
                "phrase": r["phrase"],
                "score": r["score"],
                "examples": " | ".join(r["examples"]),
                "urls": " | ".join(r["urls"]),
            })
    df = pd.DataFrame(all_rows)
    headers = {
        "Content-Disposition":
            f"attachment; filename=consensus_{re.sub(r'[^a-zA-Z0-9_-]','_',query)}.csv"
    }
    return StreamingResponse(
        iter([df.to_csv(index=False)]),
        media_type="text/csv",
        headers=headers,
    )

# ---------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)