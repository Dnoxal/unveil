#!/usr/bin/env python3
import argparse, os, sys, time, re, random
import orjson, requests
from collections import defaultdict, Counter

API_URL = "https://api.mistral.ai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You summarize user opinions about consumer products using brief, factual language.\n"
    "You receive multiple short posts about a single product.\n"
    "Extract what users think overall, the main pros and cons, and the prevailing sentiment.\n"
    "Return ONLY valid JSON with keys:\n"
    "{\n"
    '  "product": string,\n'
    '  "mentions": integer,\n'
    '  "summary": string,\n'
    '  "sentiment": "positive" | "mixed" | "negative",\n'
    '  "pros": [string],\n'
    '  "cons": [string],\n'
    '  "representative_quotes": [string],\n'
    '  "confidence": number\n'
    "}\n"
    "Be concise, avoid marketing language, and do not invent features that are not in the posts."
)

USER_TEMPLATE = """Product: {product}
Posts:
{posts}

Return JSON only with the required keys.
"""

def iter_records(path):
    # NDJSON with helpful errors; also supports JSON array
    if path.endswith(".ndjson"):
        with open(path, "rb") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    yield orjson.loads(line)
                except Exception as e:
                    print(f"JSON error on line {i}: {e}", file=sys.stderr)
                    sys.exit(1)
    else:
        with open(path, "rb") as f:
            try:
                data = orjson.loads(f.read())
            except Exception as e:
                print(f"JSON error reading array file: {e}", file=sys.stderr)
                sys.exit(1)
            if isinstance(data, list):
                for rec in data:
                    yield rec
            else:
                print("JSON file must be an array when not NDJSON", file=sys.stderr)
                sys.exit(1)

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def extract_product(rec, product_field="product"):
    # 1) preferred field (exact)
    v = rec.get(product_field)
    if isinstance(v, str) and v.strip():
        return norm_ws(v)
    # 2) common variants
    for k in ("Product", "PRODUCT", "item", "model", "name", "shampoo"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return norm_ws(v)
    # 3) light title heuristic
    title = (rec.get("title") or "").strip()
    lt = title.lower()
    if " review" in lt:
        i = lt.find(" review")
        return norm_ws(title[:i])
    if "review:" in lt:
        i = lt.find("review:")
        return norm_ws(title[i+len("review:"):])
    return ""

def build_messages(product_name: str, docs):
    joined = "\n".join(f"- {d}" for d in docs)
    user = USER_TEMPLATE.format(product=product_name, posts=joined)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

def call_mistral(model: str, messages, api_key: str, temperature=0.1, max_tokens=350, retries=6):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    backoff = 1.0
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, headers=headers, data=orjson.dumps(payload), timeout=60)
            if r.status_code == 429:
                raise RuntimeError(f"Rate limited: {r.text}")
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            try:
                return orjson.loads(content)
            except Exception:
                s = content.strip()
                if s.startswith("```"):
                    s = s.strip("`")
                    i = s.find("{")
                    if i >= 0:
                        s = s[i:]
                return orjson.loads(s)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(min(30.0, backoff))
            backoff *= 2

def sample_docs(posts_for_product, k: int, max_chars_each: int):
    sample = random.sample(posts_for_product, k) if len(posts_for_product) > k else list(posts_for_product)
    out = []
    for rec in sample:
        title = (rec.get("title") or "").strip()
        content = (rec.get("content") or rec.get("text") or "").strip()
        text = title + (" - " if title and content else "") + content
        text = norm_ws(text)
        if len(text) > max_chars_each:
            text = text[: max_chars_each - 1] + "…"
        out.append(text)
    return out

def print_table(rows):
    if not rows:
        print("No products to summarize.")
        return
    width_prod = max(7, min(40, max(len(r["product"]) for r in rows)))
    print(f"{'Rank':>4}  {'Product'.ljust(width_prod)}  {'Mentions':>8}  {'Sentiment':>9}  Summary")
    print("-" * (4 + 2 + width_prod + 2 + 8 + 2 + 9 + 2 + 60))
    for i, r in enumerate(rows, 1):
        prod = r["product"][:width_prod]
        print(f"{i:>4}  {prod.ljust(width_prod)}  {r['mentions']:>8}  {r['sentiment']:>9}  {r['summary']}")

def main():
    ap = argparse.ArgumentParser(description="Summarize what users think of each product using Mistral AI.")
    ap.add_argument("--in", dest="input_path", required=True)
    ap.add_argument("--model", default="mistral-large-latest")
    ap.add_argument("--min-mentions", type=int, default=1)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--sample", type=int, default=6)
    ap.add_argument("--max-chars-each", type=int, default=280)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--rate-sleep", type=float, default=0.0)
    ap.add_argument("--product-field", default="product")
    ap.add_argument("--out-json")
    args = ap.parse_args()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Missing MISTRAL_API_KEY", file=sys.stderr)
        sys.exit(1)

    # Group posts by product
    buckets = defaultdict(list)
    counts = Counter()
    total = 0
    for rec in iter_records(args.input_path):
        total += 1
        p = extract_product(rec, args.product_field)
        if not p:
            continue
        buckets[p].append(rec)
        counts[p] += 1

    if not counts:
        print("No product field found in input.", file=sys.stderr)
        sys.exit(1)

    # Preview what we found
    top_preview = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max(10, args.top)]
    print("Detected products (preview):")
    for name, n in top_preview:
        print(f"  - {name}  x{n}")
    print("")

    items = [(p, n) for p, n in counts.items() if n >= args.min_mentions]
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[: args.top]

    results = []
    for p, n in items:
        docs = sample_docs(buckets[p], k=args.sample, max_chars_each=args.max_chars_each)
        msgs = build_messages(p, docs)
        try:
            obj = call_mistral(args.model, msgs, api_key, temperature=args.temperature, max_tokens=400)
        except Exception as e:
            obj = {
                "product": p,
                "mentions": n,
                "summary": f"Error: {str(e)[:180]}",
                "sentiment": "mixed",
                "pros": [],
                "cons": [],
                "representative_quotes": [],
                "confidence": 0.0,
            }
        obj.setdefault("product", p)
        obj.setdefault("mentions", n)
        obj["product"] = obj["product"] or p
        obj["mentions"] = obj.get("mentions") or n
        obj.setdefault("summary", "")
        obj.setdefault("sentiment", "mixed")
        obj.setdefault("pros", [])
        obj.setdefault("cons", [])
        obj.setdefault("representative_quotes", [])
        obj.setdefault("confidence", 0.0)
        results.append(obj)
        if args.rate_sleep:
            time.sleep(args.rate_sleep)

    print(f"Scanned {total} records. Summarized {len(results)} products.\n")
    results.sort(key=lambda r: (-int(r.get("mentions", 0)), r.get("product","")))
    print_table(results)

    if args.out_json:
        with open(args.out_json, "wb") as f:
            f.write(orjson.dumps({"results": results}, option=orjson.OPT_INDENT_2))
        print(f"\nSaved JSON to {args.out_json}")

if __name__ == "__main__":
    main()
