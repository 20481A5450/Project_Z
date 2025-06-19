from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def greet_json():
    return {"Hello": "World!"}

@app.get("/system_check")
def system_check():
    return {"status": "ok", "message": "System is running smoothly!"}