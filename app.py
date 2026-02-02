# pyright: reportMissingImports=false

import json
import os
import re
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AzureOpenAI


# Load .env automatically for local development (no-op on Vercel).
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)
except Exception:
    # If python-dotenv isn't installed or .env isn't present, continue.
    pass


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


EXTRACT_PROMPT_TEMPLATE = """You are extracting compliance RULES from regulatory text.

Definition of a rule:
- An obligation, prohibition, requirement, or condition that someone must follow.
- Usually contains MUST/SHALL/REQUIRED/PROHIBITED/MAY NOT/ONLY IF, etc.
- Convert vague prose into clear, atomic rules when possible.
- Keep the meaning faithful; do not invent new requirements.

Task:
From the text below, output ONLY a JSON array of strings.
Each string must be ONE atomic rule.
No citations. No section numbers. No commentary.

Rules should be phrased as:
- "The manufacturer must ..."
- "The label must include ..."
- "Records must be retained for ..."

If the text contains no rules, output [].

TEXT:
<<<
{chunk_text}
>>>
"""

SCANNED_PDF_MESSAGE = "Scanned PDF detected (no selectable text). Digital PDFs only."


_azure_client: AzureOpenAI | None = None


def _log(step: str, **fields) -> None:
    parts = [f"extract-rules: step={step}"]
    for k, v in fields.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts))


def _get_azure_client() -> AzureOpenAI:
    global _azure_client
    if _azure_client is not None:
        return _azure_client

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    missing = [name for name, val in [("AZURE_OPENAI_API_KEY", api_key), ("AZURE_OPENAI_ENDPOINT", endpoint)] if not val]
    if missing:
        raise RuntimeError(f"Missing required Azure OpenAI env var(s): {', '.join(missing)}")

    _azure_client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
    return _azure_client


def _clean_text(text: str) -> str:
    # Normalize newlines first.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix hyphenated line breaks: "regu-\nlation" -> "regulation"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Preserve paragraph breaks (2+ newlines). Within paragraphs, replace single newlines with spaces.
    parts = re.split(r"\n{2,}", text)
    cleaned_parts: List[str] = []
    for part in parts:
        part = re.sub(r"\n+", " ", part)
        part = re.sub(r"\s+", " ", part).strip()
        if part:
            cleaned_parts.append(part)

    return "\n\n".join(cleaned_parts)


def _chunk_words(text: str, *, chunk_words: int = 800, overlap_words: int = 80) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    i = 0
    n = len(words)
    step = max(1, chunk_words - overlap_words)
    while i < n:
        j = min(n, i + chunk_words)
        chunk = " ".join(words[i:j]).strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


def _parse_rules_array(raw: str) -> List[str]:
    data = json.loads(raw)
    if not isinstance(data, list) or any(not isinstance(x, str) for x in data):
        raise ValueError("LLM output must be a JSON array of strings")
    return data


def _dedupe_stable(rules: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for r in rules:
        key = re.sub(r"\s+", " ", r).strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(re.sub(r"\s+", " ", r).strip())
    return out


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return "\n".join((page.get_text("text") or "") for page in doc)
    finally:
        doc.close()


def _llm_extract_rules_for_chunk(chunk_text: str, *, chunk_index: int) -> List[str]:
    # If creds are missing, fail clearly (no retry).
    client = _get_azure_client()
    model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")

    _log("llm_call_start", chunk_index=chunk_index, model=model, chunk_words=len(chunk_text.split()))

    base_prompt = EXTRACT_PROMPT_TEMPLATE.format(chunk_text=chunk_text)

    # First attempt: call LLM, then parse JSON. Only JSON parsing errors trigger the retry.
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": base_prompt}],
        temperature=0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        rules = _parse_rules_array(raw)
        _log("llm_call_ok", chunk_index=chunk_index, rules=len(rules))
        return rules
    except Exception:
        _log("llm_invalid_json_retry", chunk_index=chunk_index)
        retry_prompt = base_prompt + "\n\nReturn valid JSON only. Output must be a JSON array of strings."
        resp2 = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": retry_prompt}],
            temperature=0,
        )
        raw2 = (resp2.choices[0].message.content or "").strip()
        try:
            rules2 = _parse_rules_array(raw2)
            _log("llm_retry_ok", chunk_index=chunk_index, rules=len(rules2))
            return rules2
        except Exception as e2:
            _log("llm_retry_failed", chunk_index=chunk_index)
            raise RuntimeError(f"LLM returned invalid JSON for chunk index {chunk_index}") from e2


@app.post("/extract-rules")
async def extract_rules(file: UploadFile = File(...)):
    _log("request_received", filename=file.filename, content_type=file.content_type)

    # Fail fast with a clear message if required LLM env vars are missing.
    try:
        _get_azure_client()
    except RuntimeError as e:
        _log("missing_llm_env", error=str(e))
        return JSONResponse(status_code=500, content={"message": str(e)})
    _log("llm_env_ok")

    if not (file.filename or "").lower().endswith(".pdf") and file.content_type not in (None, "", "application/pdf"):
        # Keep behavior simple; still try to parse bytes if user didn't set content-type.
        pass

    pdf_bytes = await file.read()
    _log("pdf_read_ok", bytes=len(pdf_bytes))
    extracted = _extract_text_from_pdf_bytes(pdf_bytes)
    _log("pdf_text_extracted", chars=len((extracted or "")))

    if len((extracted or "").strip()) < 500:
        _log("scanned_pdf_detected", chars=len((extracted or "").strip()))
        return JSONResponse(status_code=400, content={"message": SCANNED_PDF_MESSAGE})
    _log("scanned_check_ok", chars=len((extracted or "").strip()))

    cleaned = _clean_text(extracted)
    _log("text_cleaned", chars=len(cleaned), words=len(cleaned.split()))
    cleaned = " ".join(cleaned.split()[:2400])
    _log("text_truncated", words=len(cleaned.split()))
    chunks = _chunk_words(cleaned)
    if not chunks:
        _log("no_chunks_after_cleaning")
        return JSONResponse(status_code=400, content={"message": SCANNED_PDF_MESSAGE})

    _log("chunking_done", total_chunks=len(chunks))

    all_rules: List[str] = []
    for idx, chunk in enumerate(chunks):
        try:
            chunk_rules = _llm_extract_rules_for_chunk(chunk, chunk_index=idx)
        except RuntimeError as e:
            _log("chunk_failed", chunk_index=idx, error=str(e))
            return JSONResponse(status_code=500, content={"message": str(e), "chunk_index": idx})
        all_rules.extend(chunk_rules)
        _log("chunk_done", chunk_index=idx, merged_rules_so_far=len(all_rules))

    deduped = _dedupe_stable(all_rules)
    _log("dedupe_done", before=len(all_rules), after=len(deduped))
    _log("request_complete", rules=len(deduped))
    return JSONResponse(status_code=200, content={"rules": deduped})

