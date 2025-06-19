from fastapi import FastAPI
## This is a simple FastAPI application that provides two endpoints.
app = FastAPI()

@app.get("/")
def greet_json():
    return {"Hello": "World!"}

@app.get("/system_check")
def system_check():
    return {"status": "ok", "message": "System is running smoothly!"}