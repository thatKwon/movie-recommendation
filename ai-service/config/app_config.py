from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

def create_app() -> FastAPI:
    """
    FastAPI 앱 생성 및 설정
    Creates and configures FastAPI application
    """
    app = FastAPI(
        title="Movie Recommendation AI Service",
        description="LLM 기반 영화 추천 및 분석 API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS 설정 - 백엔드 서버와 프론트엔드 허용
    allowed_origins = [
        "http://localhost:3000",      # 프론트엔드 (Next.js)
        "http://localhost:5001",      # 백엔드 (Express)
        "http://10.1.179.245:3000",   # 네트워크 프론트엔드
        "http://10.1.179.245:5001",   # 네트워크 백엔드
    ]
    
    # 환경변수에서 추가 origin 허용
    frontend_url = os.getenv("FRONTEND_URL")
    backend_url = os.getenv("BACKEND_URL")
    
    if frontend_url and frontend_url not in allowed_origins:
        allowed_origins.append(frontend_url)
    if backend_url and backend_url not in allowed_origins:
        allowed_origins.append(backend_url)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600,  # preflight 요청 캐싱 (1시간)
    )

    return app 