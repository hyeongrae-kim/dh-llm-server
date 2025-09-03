import os

MODEL_PATH = os.getenv("MODEL_PATH")
GPU_UTIL = float(os.getenv("GPU_UTIL", "0.8"))
IDLE_SECONDS = int(os.getenv("IDLE_SECONDS", "600"))  # 10분
HEALTH_ROUTE = "/health"
GENERATE_ROUTE = "/generate/"