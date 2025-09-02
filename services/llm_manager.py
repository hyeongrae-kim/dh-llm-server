import asyncio
import gc
import logging
import time
from typing import Optional

import torch
from vllm import LLM, SamplingParams

from core.settings import MODEL_PATH, GPU_UTIL, IDLE_SECONDS

logger = logging.getLogger("llm_manager")

def _now() -> float:
    return time.monotonic()

class LLMManager:
    """
    - 첫 요청 시 로딩(lazy loading)
    - 10분 유휴 시 언로드
    - 단순 동시성 제어(asyncio.Lock)
    """
    def __init__(self) -> None:
        self._llm: Optional[LLM] = None
        self._last_used: float = 0.0
        self._load_lock = asyncio.Lock()
        self._idle_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()  # 유휴 감시 종료 신호. 여러 코루틴에게 신호보낼 수 있는 브로드캐스트.

    async def start(self) -> None:
        # idle watcher 시작
        self._idle_task = asyncio.create_task(self._idle_watcher())
        logger.info("LLMManager idle watcher started.")

    async def shutdown(self) -> None:
        # 종료 시 정리
        self._stop.set()
        if self._idle_task:
            self._idle_task.cancel()
        self.unload()

    # 서버가 올라오자마자 모델을 올리지 않고, 요청이 들어올 때 로드하기 위해 모델 로드를 함수로 분리함.
    async def ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        # lock을 사용하여 코루틴이 동시에 들어오는 것 방지
        async with self._load_lock:
            if self._llm is not None:
                return
            logger.info("Loading LLM into GPU...")
            self._llm = LLM(model=MODEL_PATH, gpu_memory_utilization=GPU_UTIL)
            logger.info("LLM loaded.")

    def unload(self) -> None:
        if self._llm is None:
            return
        logger.info("Unloading LLM from GPU...")
        self._llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("LLM unloaded.")

    async def _idle_watcher(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(60)  # 1분 주기 체크
                if self._llm is not None and (_now() - self._last_used) > IDLE_SECONDS:
                    logger.info("Idle timeout exceeded. Unloading model.")
                    self.unload()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"idle_watcher error: {e}")

    def is_loaded(self) -> bool:
        return self._llm is not None

    async def generate(self, prompt: str, *, n: int, temperature: float, top_p: float,
                       repetition_penalty: float, max_tokens: int, seed: int) -> str:
        # 요청 시각 갱신
        self._last_used = _now()

        # 필요 시 로드
        await self.ensure_loaded()

        # 샘플링 파라미터 준비
        params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            seed=seed,
        )

        # vLLM generate 호출 (단일 배치/단일 출력 가정)
        assert self._llm is not None, "LLM must be loaded."
        outputs = self._llm.generate(prompt, params)
        text = outputs[0].outputs[0].text

        # 성공 처리 후 갱신
        self._last_used = _now()
        return text

# 애플리케이션 전역에서 쓸 싱글톤 매니저
llm_manager = LLMManager()