import json
from datetime import datetime
from fastapi import HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from config.app_config import create_app
from models.movie_filter import (
    MovieFilterRequest, 
    MovieFilterResponse, 
    HealthCheckResponse,
    ErrorResponse
)
from handlers.movie_filter_handler import (
    filter_movies_with_llm,
    merge_filtered_results_with_movies
)

# FastAPI 앱 생성
app = create_app()

@app.get("/", response_model=HealthCheckResponse)
async def root():
    """
    서비스 상태 확인
    Check service status
    """
    return {
        "service": "Movie Recommendation AI Service",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    헬스 체크 엔드포인트
    Health check endpoint
    """
    return {
        "service": "movie-recommendation-ai",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# ==========================================
# 영화 필터링 API
# ==========================================

@app.post("/api/filter-movies", response_model=MovieFilterResponse)
async def filter_movies(request: MovieFilterRequest):
    """
    사용자 쿼리에 따라 영화 목록을 필터링하고 우선순위를 매깁니다.
    Filters and prioritizes movie list based on user query.
    
    **Parameters:**
    - **query**: 사용자의 자연어 쿼리 (예: "비오는 날 슬픈 감정을 달래줄 영화")
                 User's natural language query
    - **movies**: 백엔드에서 받은 영화 목록 (최대 30개 권장)
                  Movie list from backend
    - **maxResults**: 반환할 최대 영화 수 (기본값: 10)
                      Maximum number of movies to return
    - **lightweightResponse**: 경량 응답 (기본값: true, AI 결과만 반환)
                               Lightweight response
    - **minScoreThreshold**: 최소 점수 기준 (기본값: 0.5, 부적합 영화 제외)
                             Minimum score threshold
    
    **Returns:**
    ```json
    {
      "movies": [
        {
          "movieId": "영화ID / Movie ID",
          "aiReason": "추천 이유 / Recommendation reason"
        }
      ]
    }
    ```
    
    **Performance:**
    - 응답 시간 / Response time: 2-3초 / 2-3 seconds
    - 응답 크기 / Response size: ~0.8KB (95% 감소 / 95% reduction)
    """
    try:
        # 1. 영화 목록을 딕셔너리로 변환
        movies_dict = [movie.model_dump() for movie in request.movies]
        
        # 2. LLM을 사용하여 필터링
        filtered_results = await filter_movies_with_llm(
            user_query=request.query,
            movies=movies_dict,
            max_results=request.maxResults or 10,
            min_score_threshold=request.minScoreThreshold if request.minScoreThreshold is not None else 0.5
        )
        
        # 3. 경량 응답 모드에 따라 처리
        if request.lightweightResponse:
            # 경량 모드: AI 결과만 반환 (movieId, aiReason)
            lightweight_movies = [
                {
                    "movieId": item.get("movieId"),
                    "aiReason": item.get("reason")
                }
                for item in filtered_results
            ]
            response_movies = lightweight_movies
        else:
            # 일반 모드: 원본 영화 정보 + AI 결과 병합
            merged_movies = await merge_filtered_results_with_movies(
                filtered_results=filtered_results,
                original_movies=movies_dict
            )
            response_movies = merged_movies
        
        # 4. 응답 생성
        response = {
            "movies": response_movies
        }
        
        print(f"✅ Movie Filtering Completed! : {len(response_movies)}movies counted... (Query: {request.query})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"영화 필터링 중 오류가 발생했습니다: {str(e)}"
        )

# ==========================================
# 에러 핸들러
# ==========================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    HTTP 예외 핸들러
    HTTP exception handler
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    일반 예외 핸들러
    General exception handler
    """
    print(f"❌ Server Error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# FastAPI가 uvicorn을 통해 실행될 때 사용
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
