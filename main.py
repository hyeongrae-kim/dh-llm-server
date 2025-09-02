import logging
import uvicorn
from fastapi import FastAPI

from api.routes import router
from services.llm_manager import llm_manager

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="vLLM FastAPI with Gpt-oss-20b")
app.include_router(router)

# 서버가 처음 켜질 때 호출됨.
@app.on_event("startup")
async def on_startup():
    await llm_manager.start()

# 서버가 종료될 때 호출됨.
@app.on_event("shutdown")
async def on_shutdown():
    await llm_manager.shutdown()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)