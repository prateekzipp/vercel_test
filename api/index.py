from fastapi import FastAPI

app = FastAPI()


@app.get("/hello")
def hello() -> str:
    return "Hello World"

