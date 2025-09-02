import logging
from fastapi import APIRouter, HTTPException
from models.schemas import QueryRequest, GenerateResponse
from services.llm_manager import llm_manager
from core.settings import HEALTH_ROUTE, GENERATE_ROUTE

logger = logging.getLogger("api")

router = APIRouter()

@router.get(HEALTH_ROUTE)
async def health():
    return {"ok": True, "loaded": llm_manager.is_loaded()}

@router.post(GENERATE_ROUTE, response_model=GenerateResponse)
async def generate_response(request: QueryRequest):
    try:
        text = await llm_manager.generate(
            request.query,
            n=request.n,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=1.1,
            max_tokens=request.max_tokens,
            seed=request.seed,
        )
        return GenerateResponse(response=text)
    except Exception as e:
        logger.exception("Generation failed.")
        raise HTTPException(status_code=500, detail=str(e))