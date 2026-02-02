from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow v0.dev (and any origin) to talk to your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; you can restrict this later
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/hello")
def hello() -> str:
    return "Hello World"

