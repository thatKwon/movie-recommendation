"""
영화 필터링 API를 위한 Request/Response 모델
Request/Response models for Movie Filtering API
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class CastMember(BaseModel):
    """
    출연진 정보
    Cast member information
    """
    actorName: str
    character: Optional[str] = None
    order: Optional[int] = None
    profileUrl: Optional[str] = None


class Director(BaseModel):
    """
    감독 정보
    Director information
    """
    directorName: str
    profileUrl: Optional[str] = None


class MovieInfo(BaseModel):
    """
    영화 정보 모델
    Movie information model
    """
    id: str | int = Field(..., description="영화 ID (MongoDB ObjectId 또는 TMDB ID)")
    tmdbId: Optional[int] = Field(None, description="TMDB ID")
    title: str = Field(..., description="영화 제목 (한글)")
    year: Optional[int] = Field(None, description="개봉 연도")
    genres: List[str] = Field(default=[], description="장르 목록")
    plot: Optional[str] = Field(None, description="줄거리 (한글)")
    posterUrl: Optional[str] = Field(None, description="포스터 URL")
    backdropUrl: Optional[str] = Field(None, description="배경 이미지 URL")
    rating: Optional[str] = Field(None, description="관람 등급 (예: 15세 관람가)")
    voteAverage: Optional[float] = Field(None, description="평점 (0-10)")
    runtime: Optional[int] = Field(None, description="상영 시간 (분)")
    cast: Optional[List[CastMember]] = Field(None, description="출연진 정보")
    directors: Optional[List[Director]] = Field(None, description="감독 정보 목록")
    likeCount: Optional[int] = Field(None, description="좋아요 수 (백엔드 전용, LLM에는 전송하지 않음)")
    viewCount: Optional[int] = Field(None, description="조회수 (백엔드 전용, LLM에는 전송하지 않음)")
    relevanceScore: Optional[float] = Field(None, description="백엔드 추천 점수 (백엔드 알고리즘 계산, LLM에는 전송하지 않음)")


class MovieFilterRequest(BaseModel):
    """
    영화 필터링 요청 모델
    Movie filtering request model
    """
    query: str = Field(
        ..., 
        description="사용자의 자연어 쿼리",
        min_length=1,
        max_length=500,
        examples=["우주를 배경으로 한 감동적인 영화", "가족과 함께 볼 수 있는 코미디"]
    )
    movies: List[MovieInfo] = Field(
        ..., 
        description="필터링할 영화 목록",
        min_items=1,
        max_items=100
    )
    maxResults: Optional[int] = Field(
        default=10,
        description="반환할 최대 영화 수",
        ge=1,
        le=50
    )
    lightweightResponse: Optional[bool] = Field(
        default=True,
        description="경량 응답 모드 (true: AI 결과만 반환, false: 전체 영화 데이터 반환)"
    )
    minScoreThreshold: Optional[float] = Field(
        default=0.5,
        description="최소 매칭 점수 기준 (이 값 미만은 제외, 0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    @validator('query')
    def validate_query(cls, v):
        """쿼리 검증"""
        if not v or not v.strip():
            raise ValueError("쿼리는 비어있을 수 없습니다.")
        return v.strip()
    
    @validator('movies')
    def validate_movies(cls, v):
        """영화 목록 검증"""
        if not v or len(v) == 0:
            raise ValueError("영화 목록은 비어있을 수 없습니다.")
        return v


class FilteredMovie(BaseModel):
    """
    필터링된 영화 결과 (LLM 분석 포함)
    Filtered movie result (with LLM analysis)
    """
    movieId: str | int = Field(..., description="영화 ID")
    matchScore: float = Field(..., description="사용자 쿼리 부합도 (0.0-1.0)", ge=0.0, le=1.0)
    reason: str = Field(..., description="추천 이유", min_length=1)


class AIFilterResult(BaseModel):
    """
    AI 필터링 결과 (경량)
    AI filtering result (lightweight)
    """
    movieId: str | int = Field(..., description="영화 ID")
    aiReason: str = Field(..., description="AI 추천 이유")


class MovieFilterResponse(BaseModel):
    """
    영화 필터링 응답 모델
    Movie filtering response model
    """
    movies: List[Dict[str, Any]] | List[AIFilterResult] = Field(
        ..., 
        description="필터링 결과 (경량 모드: AI 결과만, 일반 모드: 전체 영화 데이터)"
    )


class HealthCheckResponse(BaseModel):
    """
    헬스 체크 응답
    Health check response
    """
    service: str
    status: str
    version: str
    timestamp: str


class ErrorResponse(BaseModel):
    """
    에러 응답
    Error response
    """
    error: str
    detail: Optional[str] = None
    timestamp: Optional[str] = None

