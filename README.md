# FastAPI on Vercel (Hello World)

This folder is a minimal FastAPI app that Vercel can deploy directly.

## Local run (uvicorn)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.index:app --reload
```

Then open `http://127.0.0.1:8000/hello`.

## Local run (Vercel)

```bash
npm i -g vercel
vercel dev
```

Your app will be available at `http://localhost:3000/`.

