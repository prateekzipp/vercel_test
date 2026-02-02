# FastAPI PDF → Rules Extractor

This folder is a minimal FastAPI app with one endpoint that accepts a **digital PDF** and extracts compliance rules using an LLM.

## Local run (uvicorn)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Put Azure env vars in `vercel_test/.env` (loaded automatically on startup)
uvicorn app:app --reload
```

Then call the endpoint:

```bash
curl -F "file=@reg.pdf" http://localhost:8000/extract-rules
```

## Required env vars (Azure OpenAI)

Set these in your shell, `.env` (locally), and in Vercel project env vars later:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION` (optional, default: `2024-02-15-preview`)
- `AZURE_OPENAI_MODEL` (optional, default: `gpt-4o`)

## Local run (Vercel)

```bash
npm i -g vercel
vercel dev
```

Your app will be available at `http://localhost:3000/`.

