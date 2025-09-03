# DH LLM API Server

이 프로젝트는 로컬 서버에서 LLM(Open-Source Model)을 실행하고 REST API 형태로 제공하기 위한 서버입니다.  
경북대학교 디지털인문공학연구소의 내부 서버에서 사용하기 위해 만들어졌습니다.

## 환경 설정

프로젝트 루트 폴더에 `.env` 파일을 생성하여 아래와 같이 설정합니다:

```bash
MODEL_PATH=/home/on_premise/models/gpt-oss-20b
GPU_UTIL=0.8
IDLE_SECONDS=600
```

### 환경 변수 설명

- `MODEL_PATH`: 사용할 모델의 경로
- `GPU_UTIL`: GPU 최대 활용 비율 (0.0 ~ 1.0)
- `IDLE_SECONDS`: 유휴 상태 시 자동 종료까지의 대기 시간(초)

## 실행 방법

아래 명령어를 실행하면 서버가 8000 포트에서 실행됩니다:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 > server.log 2>&1
```

로그는 `server.log` 파일에 저장됩니다.

## 종료 방법

실행 중인 터미널에서 `Ctrl + C`를 눌러 서버를 종료합니다.