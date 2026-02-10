# pyright: reportMissingImports=false

import base64
import json
import os
import re
import time
import requests
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Optional

import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AzureOpenAI
from supabase import create_client, Client
from pydantic import BaseModel


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


ALLOWED_CATEGORIES = [
    "General rules",
    "PDP (Prinicipal Display Panel)",
    "Ingredients",
    "Nutrition",
    "Information to Include (Warnings)",
    "Allergen",
    "Location/Source",
    "Max permissible limits",
]


class ExtractedRule(TypedDict):
    rule: str
    category: str


class ModuleIdsRequest(BaseModel):
    module_ids: List[str]


EXTRACT_PROMPT_TEMPLATE = """You are extracting compliance RULES from regulatory text.

Definition of a rule:
- An obligation, prohibition, requirement, or condition that someone must follow.
- Usually contains MUST/SHALL/REQUIRED/PROHIBITED/MAY NOT/ONLY IF, etc.
- Convert vague prose into clear, atomic rules when possible.
- Keep the meaning faithful; do not invent new requirements.

Task:
From the text below, output ONLY a JSON array of objects.
Each object must have exactly:
- "rule": string (ONE atomic rule)
- "category": string (choose exactly one from the allowed list below)

Allowed categories (must match exactly):
- General rules
- PDP (Prinicipal Display Panel)
- Ingredients
- Nutrition
- Information to Include (Warnings)
- Allergen
- Location/Source
- Max permissible limits

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

COMPLIANCE_CHECK_PROMPT = """You are a compliance verification expert analyzing product images against regulatory rules.

PRODUCT ATTRIBUTES:
{product_attributes}

COMPLIANCE RULES TO CHECK:
{rules_list}

TASK:
For each rule listed above, you must perform TWO checks:

1. APPLICABILITY CHECK:
   - If "Applicability" is NULL or empty, the rule ALWAYS applies to this product
   - If "Applicability" has text, determine if it semantically matches the Product Attributes
   - Use semantic understanding (not exact text matching)

2. COMPLIANCE CHECK (only if rule applies):
   - Analyze ALL provided product images
   - Determine if the product complies with the rule's instruction

For each rule, provide:
1. "status": Must be exactly one of:
   - "True" (rule applies AND product is compliant)
   - "False" (rule applies AND product is NOT compliant)
   - "Not Applicable" (rule does not apply to this product based on applicability check)
2. "reasoning": Brief explanation covering:
   - WHY the rule applies or doesn't apply (if applicability text exists)
   - Whether the product is compliant (if rule applies)
   - Which image(s) you examined

IMPORTANT:
- A rule is "True" if it applies AND ANY of the images satisfies the requirement
- A rule is "False" if it applies AND you cannot confirm compliance
- A rule is "Not Applicable" ONLY if the applicability text doesn't match the product attributes
- If images are unclear but rule applies, mark as "False" and explain in reasoning

OUTPUT FORMAT:
Return ONLY a valid JSON array with this exact structure:
[
  {{
    "rule_id": "uuid-here",
    "status": "True|False|Not Applicable",
    "reasoning": "explanation here"
  }}
]
"""

SCANNED_PDF_MESSAGE = "Scanned PDF detected (no selectable text). Digital PDFs only."


_azure_client: AzureOpenAI | None = None
_supabase_client: Client | None = None


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


def _get_supabase_client() -> Client:
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise RuntimeError("Missing required Supabase env var(s): SUPABASE_URL, SUPABASE_SERVICE_KEY")

    _supabase_client = create_client(url, key)
    return _supabase_client


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


def _parse_rules_array(raw: str) -> List[ExtractedRule]:
    s = (raw or "").strip()

    # Common model behavior: wrap JSON in markdown fences.
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```$", "", s).strip()

    def _load_json_array(text: str) -> Any:
        # First try: parse as-is.
        try:
            return json.loads(text)
        except Exception:
            # Second try: extract the first [...] block.
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    data: Any = _load_json_array(s)
    if not isinstance(data, list):
        raise ValueError("LLM output must be a JSON array")

    allowed = set(ALLOWED_CATEGORIES)
    out: List[ExtractedRule] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("LLM output items must be objects")
        rule = item.get("rule")
        category = item.get("category")
        if not isinstance(rule, str) or not isinstance(category, str):
            raise ValueError('Each item must have "rule" and "category" strings')
        rule_clean = re.sub(r"\s+", " ", rule).strip()
        if not rule_clean:
            continue
        if category not in allowed:
            raise ValueError(f"Invalid category: {category!r}")
        out.append({"rule": rule_clean, "category": category})
    return out


def _dedupe_stable(rules: List[ExtractedRule]) -> List[ExtractedRule]:
    seen: set[str] = set()
    out: List[ExtractedRule] = []
    for r in rules:
        key = re.sub(r"\s+", " ", r["rule"]).strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"rule": re.sub(r"\s+", " ", r["rule"]).strip(), "category": r["category"]})
    return out


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return "\n".join((page.get_text("text") or "") for page in doc)
    finally:
        doc.close()


def _encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string for OpenAI vision API."""
    return base64.b64encode(image_bytes).decode('utf-8')


def _parse_compliance_results(raw: str) -> List[Dict[str, Any]]:
    """Parse LLM JSON response for compliance results."""
    s = raw.strip()
    
    # Remove markdown fences if present
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```$", "", s).strip()
    
    try:
        data = json.loads(s)
    except:
        # Try extracting first [...] block
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1:
            data = json.loads(s[start:end+1])
        else:
            raise ValueError("Could not parse LLM response as JSON")
    
    if not isinstance(data, list):
        raise ValueError("Expected JSON array from LLM")
    
    return data


def _check_compliance_with_llm(
    client: AzureOpenAI,
    images: List[bytes],
    product_attributes: str,
    rules: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Call Azure OpenAI vision model to check compliance for all rules.
    Returns list of compliance results per rule.
    """
    model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
    
    # Format rules for prompt - include applicability_text
    rules_formatted = []
    for idx, rule in enumerate(rules, 1):
        logic_config = rule.get("logic_config", {})
        instruction = logic_config.get("instruction", "No instruction provided")
        applicability = rule.get("applicability_text") or "NULL"
        
        rules_formatted.append(
            f"{idx}. Rule ID: {rule['id']}\n"
            f"   Rule Code: {rule.get('rule_code', 'N/A')}\n"
            f"   Applicability: {applicability}\n"
            f"   Instruction: {instruction}"
        )
    
    rules_text = "\n\n".join(rules_formatted)
    prompt = COMPLIANCE_CHECK_PROMPT.format(
        product_attributes=product_attributes,
        rules_list=rules_text
    )
    
    # Build message with text + images
    message_content = [{"type": "text", "text": prompt}]
    
    for img_bytes in images:
        base64_image = _encode_image_to_base64(img_bytes)
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    # Call LLM
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message_content}],
        temperature=0,
        max_tokens=4000
    )
    
    raw_response = response.choices[0].message.content or ""
    
    # Parse JSON response
    parsed = _parse_compliance_results(raw_response)
    return parsed


def _llm_extract_rules_for_chunk(chunk_text: str, *, chunk_index: int) -> List[ExtractedRule]:
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
    except Exception as e:
        _log("llm_invalid_json_retry", chunk_index=chunk_index, error=type(e).__name__)
        retry_prompt = (
            base_prompt
            + "\n\nReturn valid JSON only. Output must be a JSON array of objects with keys: "
            + '"rule" (string) and "category" (string from the allowed list).'
        )
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
            _log("llm_retry_failed", chunk_index=chunk_index, error=type(e2).__name__)
            raise RuntimeError(
                f"LLM returned invalid JSON for chunk index {chunk_index}: {e2}"
            ) from e2


@app.get("/regulations")
async def list_regulations():
    """List all regulatory frameworks from the regulatory_frameworks table."""
    try:
        client = _get_supabase_client()
        response = client.table("regulatory_frameworks").select("*").execute()
        return JSONResponse(status_code=200, content={"regulations": response.data})
    except RuntimeError as e:
        _log("supabase_env_error", error=str(e))
        return JSONResponse(status_code=500, content={"message": str(e)})
    except Exception as e:
        _log("supabase_query_error", error=str(e))
        return JSONResponse(status_code=500, content={"message": f"Database error: {str(e)}"})


@app.get("/regulations/{regulation_id}/engines")
async def get_regulation_engines(regulation_id: str):
    """Get all ENGINE type compliance modules for a specific regulatory framework."""
    try:
        client = _get_supabase_client()
        response = (
            client.table("compliance_modules")
            .select("*")
            .eq("framework_id", regulation_id)
            .eq("module_type", "ENGINE")
            .execute()
        )
        return JSONResponse(status_code=200, content={"engines": response.data})
    except RuntimeError as e:
        _log("supabase_env_error", error=str(e))
        return JSONResponse(status_code=500, content={"message": str(e)})
    except Exception as e:
        _log("supabase_query_error", error=str(e))
        return JSONResponse(status_code=500, content={"message": f"Database error: {str(e)}"})


def _get_module_tree_recursive(client: Any, parent_id: str, visited: set) -> List[Dict[str, Any]]:
    """Recursively fetch all child modules starting from a parent_id."""
    if parent_id in visited:
        return []
    
    visited.add(parent_id)
    
    # Fetch all modules with this parent_id
    response = client.table("compliance_modules").select("*").eq("parent_id", parent_id).execute()
    
    children = response.data or []
    all_descendants = list(children)
    
    # Recursively fetch descendants for each child
    for child in children:
        child_id = child.get("id")
        if child_id:
            descendants = _get_module_tree_recursive(client, child_id, visited)
            all_descendants.extend(descendants)
    
    return all_descendants


@app.get("/engines/{engine_id}/modules")
async def get_engine_modules(engine_id: str):
    """Get all compliance modules reachable from a given ENGINE module (recursive tree traversal)."""
    try:
        client = _get_supabase_client()
        
        # First, fetch the engine module itself
        engine_response = client.table("compliance_modules").select("*").eq("id", engine_id).execute()
        
        if not engine_response.data:
            return JSONResponse(status_code=404, content={"message": f"Engine module {engine_id} not found"})
        
        engine = engine_response.data[0]
        
        # Recursively fetch all descendants
        visited: set = set()
        descendants = _get_module_tree_recursive(client, engine_id, visited)
        
        # Return the engine plus all its descendants
        all_modules = [engine] + descendants
        
        # Extract just the IDs
        module_ids = [module.get("id") for module in all_modules if module.get("id")]
        
        return JSONResponse(
            status_code=200, 
            content={
                "engine": engine,
                "total_modules": len(all_modules),
                "module_ids": module_ids,
                "modules": all_modules
            }
        )
    except RuntimeError as e:
        _log("supabase_env_error", error=str(e))
        return JSONResponse(status_code=500, content={"message": str(e)})
    except Exception as e:
        _log("supabase_query_error", error=str(e))
        return JSONResponse(status_code=500, content={"message": f"Database error: {str(e)}"})


@app.post("/modules/rules")
async def get_rules_by_modules(request: ModuleIdsRequest):
    """Get all compliance rules for a list of module IDs."""
    try:
        client = _get_supabase_client()
        module_ids = request.module_ids
        
        if not module_ids:
            return JSONResponse(status_code=400, content={"message": "module_ids list cannot be empty"})
        
        # Query compliance_rules where module_id is in the provided list
        response = client.table("compliance_rules").select("*").in_("module_id", module_ids).execute()
        
        rules = response.data or []
        
        return JSONResponse(
            status_code=200,
            content={
                "total_rules": len(rules),
                "module_count": len(module_ids),
                "rules": rules
            }
        )
    except RuntimeError as e:
        _log("supabase_env_error", error=str(e))
        return JSONResponse(status_code=500, content={"message": str(e)})
    except Exception as e:
        _log("supabase_query_error", error=str(e))
        return JSONResponse(status_code=500, content={"message": f"Database error: {str(e)}"})


@app.post("/check-compliance")
async def check_compliance(
    engine_id: str = Body(...),
    product_attributes: str = Body(...),
    file: UploadFile = File(...),
    save_to_db: bool = Body(False)
):
    """
    Extract fields from a product label PDF using Azure Content Understanding.
    
    Args:
        engine_id: UUID of the compliance engine
        product_attributes: Text description of the product (e.g., "Mango juice for children")
        file: Multi-page PDF file of the product label
        save_to_db: Whether to save results to product_compliance_status table (default: False)
    
    Returns:
        Extracted fields with values, bounding boxes, and metadata along with applicable rules
    """
    _log("field_extraction_start", engine_id=engine_id, filename=file.filename, save_to_db=save_to_db)
    
    try:
        # 1. Get Supabase client
        supabase_client = _get_supabase_client()
        
        # 2. Fetch engine and all its modules
        engine_response = supabase_client.table("compliance_modules").select("*").eq("id", engine_id).execute()
        if not engine_response.data:
            return JSONResponse(status_code=404, content={"message": f"Engine {engine_id} not found"})
        
        # Get all descendant modules
        visited: set = set()
        descendants = _get_module_tree_recursive(supabase_client, engine_id, visited)
        all_modules = engine_response.data + descendants
        module_ids = [m["id"] for m in all_modules if m.get("id")]
        
        _log("modules_fetched", count=len(module_ids))
        
        # 3. Fetch all rules for these modules
        if not module_ids:
            return JSONResponse(status_code=400, content={"message": "No modules found for engine"})
        
        rules_response = supabase_client.table("compliance_rules").select("*").in_("module_id", module_ids).execute()
        rules = rules_response.data or []
        
        _log("rules_fetched", total=len(rules))
        
        # 4. Read the PDF file
        pdf_bytes = await file.read()
        _log("pdf_read", bytes=len(pdf_bytes))
        
        # 5. Analyze PDF with Azure Content Understanding to extract fields
        extracted_fields = _analyze_product_document(pdf_bytes)
        if extracted_fields:
            _log("product_fields_extracted", 
                 num_contents=len(extracted_fields.get("contents", [])))
            
            # 6. Match rule extraction_fields with extracted data
            enriched_rules = _match_rule_fields_with_extracted_data(rules, extracted_fields)
            
            _log("fields_matched", total_rules=len(enriched_rules))
            
            # 7. Optionally save results to database
            rows_saved = 0
            if save_to_db:
                try:
                    rows_saved = _save_compliance_status_to_db(
                        supabase_client, 
                        enriched_rules
                    )
                except Exception as e:
                    _log("db_save_error", error=str(e))
                    # Don't fail the entire request if DB save fails
            
            return JSONResponse(status_code=200, content={
                "engine_id": engine_id,
                "product_attributes": product_attributes,
                "filename": file.filename,
                "total_rules": len(enriched_rules),
                "rules": enriched_rules,
                "extracted_fields": extracted_fields,  # Keep full extraction for reference
                "saved_to_db": save_to_db,
                "rows_saved": rows_saved
            })
        else:
            _log("product_fields_extraction_failed")
            return JSONResponse(status_code=500, content={"message": "Failed to extract fields from PDF"})
        
    except RuntimeError as e:
        _log("field_extraction_error", error=str(e))
        return JSONResponse(status_code=500, content={"message": str(e)})
    except Exception as e:
        _log("field_extraction_error", error=str(e))
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})


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

    all_rules: List[ExtractedRule] = []
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


def _get_content_understanding_config() -> dict:
    """Get Azure Content Understanding configuration from environment variables."""
    resource_name = os.getenv("AZURE_CONTENT_UNDERSTANDING_RESOURCE_NAME")
    api_key = os.getenv("AZURE_CONTENT_UNDERSTANDING_KEY")
    analyzer_id = os.getenv("AZURE_CONTENT_UNDERSTANDING_ANALYZER_ID")
    api_version = os.getenv("AZURE_CONTENT_UNDERSTANDING_API_VERSION", "2025-11-01")
    
    missing = []
    if not resource_name:
        missing.append("AZURE_CONTENT_UNDERSTANDING_RESOURCE_NAME")
    if not api_key:
        missing.append("AZURE_CONTENT_UNDERSTANDING_KEY")
    if not analyzer_id:
        missing.append("AZURE_CONTENT_UNDERSTANDING_ANALYZER_ID")
    
    if missing:
        raise RuntimeError(f"Missing required Azure Content Understanding env var(s): {', '.join(missing)}")
    
    endpoint = f"https://{resource_name}.cognitiveservices.azure.com"
    
    return {
        "endpoint": endpoint,
        "api_key": api_key,
        "analyzer_id": analyzer_id,
        "api_version": api_version
    }


def _submit_document_analysis(pdf_bytes: bytes, config: dict) -> str | None:
    """
    Submit a PDF document to Azure Content Understanding for analysis.
    Returns the result_id for polling, or None on error.
    """
    analyze_url = (
        f"{config['endpoint']}/contentunderstanding/analyzers/"
        f"{config['analyzer_id']}:analyzeBinary?api-version={config['api_version']}"
    )
    
    headers = {
        "Ocp-Apim-Subscription-Key": config["api_key"],
        "Content-Type": "application/octet-stream"
    }
    
    _log("content_understanding_submit", bytes_size=len(pdf_bytes))
    
    try:
        response = requests.post(analyze_url, headers=headers, data=pdf_bytes, timeout=30)
        response.raise_for_status()
        result_id = response.json()["id"]
        _log("content_understanding_submitted", result_id=result_id)
        return result_id
    except requests.exceptions.RequestException as e:
        _log("content_understanding_submit_error", error=str(e))
        return None


def _poll_analysis_results(result_id: str, config: dict, max_wait_seconds: int = 120) -> dict | None:
    """
    Poll for analysis results until completion or timeout.
    Returns the final result data or None on failure.
    """
    result_url = (
        f"{config['endpoint']}/contentunderstanding/analyzerResults/"
        f"{result_id}?api-version={config['api_version']}"
    )
    
    headers = {
        "Ocp-Apim-Subscription-Key": config["api_key"]
    }
    
    _log("content_understanding_polling_start", result_id=result_id)
    
    start_time = time.time()
    poll_interval = 2  # seconds
    
    while (time.time() - start_time) < max_wait_seconds:
        time.sleep(poll_interval)
        
        try:
            response = requests.get(result_url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            status = data.get("status")
            
            _log("content_understanding_poll", status=status)
            
            if status == "Succeeded":
                _log("content_understanding_complete", result_id=result_id)
                return data
            elif status in ["Failed", "Canceled"]:
                _log("content_understanding_failed", status=status, result_id=result_id)
                return None
                
        except requests.exceptions.RequestException as e:
            _log("content_understanding_poll_error", error=str(e))
            return None
    
    _log("content_understanding_timeout", result_id=result_id)
    return None


def _analyze_product_document(pdf_bytes: bytes) -> dict | None:
    """
    Analyze a product document using Azure Content Understanding.
    Returns extracted fields data or None on error.
    
    Returns structure:
    {
        "contents": [
            {
                "fields": {
                    "field_name": {
                        "value": "...",
                        "valueString": "...",
                        "boundingRegions": [...],
                        ...
                    }
                }
            }
        ]
    }
    """
    try:
        config = _get_content_understanding_config()
    except RuntimeError as e:
        _log("content_understanding_config_error", error=str(e))
        return None
    
    # Submit document for analysis
    result_id = _submit_document_analysis(pdf_bytes, config)
    if not result_id:
        return None
    
    # Poll for results
    analysis_result = _poll_analysis_results(result_id, config)
    if not analysis_result:
        return None
    
    # Extract the result contents
    return analysis_result.get("result", {})

def _save_compliance_status_to_db(
    supabase_client: Client,
    enriched_rules: List[Dict[str, Any]],
    product_id: str = "426fc891-7c40-493c-b4d0-cfd0ec9d539f"
) -> int:
    """
    Save compliance status results to product_compliance_status table.
    
    Args:
        supabase_client: Supabase client instance
        enriched_rules: List of rules with matched extraction_field data
        product_id: Fixed product ID for all rows
    
    Returns:
        Number of rows inserted
    """
    rows_to_insert = []
    
    for rule in enriched_rules:
        rule_id = rule.get("id")
        extraction_field = rule.get("extraction_field", [])
        
        # Determine compliance status and reason
        missing_fields = []
        found_fields = []
        
        for field_data in extraction_field:
            field_name = field_data.get("field_name")
            value = field_data.get("value")
            
            # Check if field has an error or null value
            if field_data.get("error") or value is None:
                missing_fields.append(field_name)
            else:
                found_fields.append(field_name)
        
        # Determine compliance status
        if missing_fields:
            compliance_status = "FAIL"
            # Create compliance reason message
            compliance_reason = ", ".join([f"{field} not found" for field in missing_fields])
        else:
            compliance_status = "PASS"
            compliance_reason = "All required fields found"
        
        # Prepare row for insertion
        rows_to_insert.append({
            "product_id": product_id,
            "rule_id": rule_id,
            "is_applicable": True,
            "compliance_status": compliance_status,
            "compliance_reason": compliance_reason,
            "is_active": True,
            "extraction_values": extraction_field  # Store the full enriched extraction_field
        })
    
    # Insert all rows
    if rows_to_insert:
        response = supabase_client.table("product_compliance_status").insert(rows_to_insert).execute()
        _log("compliance_status_saved", rows_inserted=len(rows_to_insert))
        return len(rows_to_insert)
    
    return 0


def _match_rule_fields_with_extracted_data(
    rules: List[Dict[str, Any]], 
    extracted_fields: dict
) -> List[Dict[str, Any]]:
    """
    Match each rule's extraction_field with extracted fields from Azure Content Understanding.
    Replace extraction_field array with full field data including bounding boxes.
    
    Args:
        rules: List of compliance rules from database
        extracted_fields: Extracted fields from Azure Content Understanding
    
    Returns:
        List of enriched rules with full field data from Azure
    """
    # Extract all fields from Azure response
    azure_fields = {}
    for content in extracted_fields.get("contents", []):
        fields = content.get("fields", {})
        azure_fields.update(fields)
    
    _log("azure_fields_available", fields=list(azure_fields.keys()))
    
    enriched_rules = []
    
    for rule in rules:
        # Get the extraction_field array from the rule (note: singular, not plural)
        original_extraction_field = rule.get("extraction_field", [])
        
        if not original_extraction_field:
            # Rule has no extraction_field specified - keep as is
            enriched_rules.append(rule)
            continue
        
        # Build new extraction_field with full data from Azure
        enriched_extraction_field = []
        
        # Match each field name with Azure extracted fields
        for field_name in original_extraction_field:
            if field_name in azure_fields:
                field_data = azure_fields[field_name]
                
                # Extract all relevant information from Azure output
                # Note: source contains the bounding box data in Azure's format
                enriched_extraction_field.append({
                    "field_name": field_name,
                    "value": field_data.get("value") or field_data.get("valueString"),
                    "valueString": field_data.get("valueString"),
                    "confidence": field_data.get("confidence"),
                    "source": field_data.get("source"),  # This is the bounding box data
                    "spans": field_data.get("spans", []),
                    "type": field_data.get("type")
                })
            else:
                # Field not found in Azure extraction - indicate as null
                enriched_extraction_field.append({
                    "field_name": field_name,
                    "value": None,
                    "valueString": None,
                    "confidence": None,
                    "source": None,
                    "spans": [],
                    "error": "Field not found in extracted data"
                })
        
        # Replace extraction_field with enriched version
        enriched_rule = {**rule}
        enriched_rule["extraction_field"] = enriched_extraction_field
        enriched_rules.append(enriched_rule)
    
    return enriched_rules
