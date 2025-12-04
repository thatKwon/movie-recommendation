# AI Service - 영화 추천 필터링 API

LLM 기반 영화 필터링 서비스. 백엔드 추천 알고리즘의 후보 영화 목록을 사용자의 자연어 요청에 맞게 재필터링합니다.

## 기술 스택

- **Framework**: FastAPI
- **LLM**: OpenAI GPT-4o-mini (LangChain)
- **Monitoring**: LangSmith

## 프로젝트 구조

```
ai-service/
├── main.py                    # FastAPI 엔트리포인트
├── config/
│   ├── app_config.py          # FastAPI 앱 설정
│   └── service_config.py      # LLM/API 키 설정
├── handlers/
│   └── movie_filter_handler.py  # 영화 필터링 로직
├── models/
│   └── movie_filter.py        # Request/Response 모델
├── load/
│   └── prompt_loader.py       # YAML 프롬프트 로더
└── resources/
    └── movie_filter_prompt.yaml  # LLM 프롬프트
```

## API 엔드포인트

### `POST /api/filter-movies`

영화 목록을 사용자 쿼리에 맞게 필터링합니다.

**Request:**
```json
{
  "query": "비오는 날 슬픈 감정을 달래줄 영화",
  "movies": [
    {
      "id": "673f7c8e9d8f2a1b3c4d5ea1",
      "title": "포레스트 검프",
      "year": 1994,
      "genres": ["드라마", "로맨스"],
      "plot": "순수한 마음을 가진 포레스트 검프가...",
      "voteAverage": 8.5
    }
  ],
  "maxResults": 10,
  "lightweightResponse": true
}
```

**Response:**
```json
{
  "movies": [
    {
      "movieId": "673f7c8e9d8f2a1b3c4d5ea1",
      "aiReason": "따뜻하고 희망적인 휴먼 드라마."
    }
  ]
}
```

### 기타 엔드포인트

| 엔드포인트 | 설명 |
|-----------|------|
| `GET /` | 서비스 상태 확인 |
| `GET /health` | 헬스 체크 |
| `GET /docs` | Swagger UI |

## 설치 및 실행

### 1. 가상환경 생성 및 의존성 설치

```bash
cd ai-service
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 환경변수 설정 (.env 파일 생성)
OPENAI_API_KEY=sk-your-key
LANGSMITH_API_KEY=lsv2_pt_your-key

```bash
# 서버 오픈
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 서버 로그 확인
tail -f nohup.out
```

## 성능

| 항목 | 값 |
|-----|-----|
| 응답 시간 | 2-3초 |
| 응답 크기 | ~0.8KB |
| 모델 | gpt-4o-mini |
| 포트 | 8000 |
