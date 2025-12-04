"""
영화 필터링 핸들러
Movie Filtering Handler

백엔드에서 받은 영화 목록을 사용자 쿼리에 맞게 필터링하고 우선순위를 매깁니다.
Filters and prioritizes movie list from backend based on user query.
"""

import json
import os
from typing import List, Dict, Any, Optional
from fastapi import HTTPException
from langchain.schema import HumanMessage, SystemMessage

from config.service_config import llm
from load import load_prompt

# 프롬프트 파일 경로
PROMPT_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "resources",
    "movie_filter_prompt.yaml"
)


def prepare_movies_for_llm(movies: List[Dict[str, Any]]) -> str:
    """
    영화 목록을 LLM이 이해하기 쉬운 형식으로 변환합니다.
    Converts movie list into LLM-friendly format.
    
    Args:
        movies: 백엔드에서 받은 영화 목록 / Movie list from backend
        
    Returns:
        str: JSON 형식의 영화 정보 문자열 / Movie information string in JSON format
    """
    simplified_movies = []
    
    for movie in movies:
        # 감독 이름 추출 (객체 배열 → 문자열 배열)
        directors_list = movie.get("directors", [])
        if directors_list and isinstance(directors_list[0], dict):
            directors = [d.get("directorName", "") for d in directors_list if d.get("directorName")]
        else:
            directors = directors_list
        
        # 배우 정보 간소화 (상위 3명만, 이름과 역할만)
        cast_list = movie.get("cast", [])
        if cast_list:
            cast = [
                {
                    "actorName": c.get("actorName", ""),
                    "character": c.get("character", "")
                }
                for c in cast_list[:3]
                if c.get("actorName")
            ]
        else:
            cast = []
        
        # 핵심 정보만 포함
        simplified = {
            "id": movie.get("id"),
            "title": movie.get("title"),
            "year": movie.get("year"),
            "genres": movie.get("genres", []),
            "plot": movie.get("plot", "")[:200] if movie.get("plot") else "",
            "voteAverage": movie.get("voteAverage", 0),
            "directors": directors[:1] if directors else []
        }
        
        # 빈 값 제거 (선택적)
        simplified = {k: v for k, v in simplified.items() if v or v == 0}
        simplified_movies.append(simplified)
    
    return json.dumps(simplified_movies, ensure_ascii=False, indent=2)


async def create_filter_prompt(user_query: str, movies_json: str) -> List[Any]:
    """
    영화 필터링을 위한 프롬프트를 생성합니다.
    Creates prompt for movie filtering.
    
    YAML 파일에서 프롬프트를 로드합니다.
    Loads prompt from YAML file.
    
    Args:
        user_query: 사용자의 자연어 쿼리 / User's natural language query
        movies_json: JSON 형식의 영화 목록 / Movie list in JSON format
        
    Returns:
        List: LangChain 메시지 리스트 / LangChain message list
    """
    # YAML 파일에서 프롬프트 로드
    prompt_template = await load_prompt(PROMPT_FILE_PATH)
    
    # 시스템 프롬프트
    system_prompt = prompt_template.get('system', '')
    
    # 사용자 프롬프트 (템플릿 변수 치환)
    input_template = prompt_template.get('input', '')
    user_prompt = input_template.format(
        user_query=user_query,
        movies_json=movies_json
    )

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]


def parse_filter_response(response_text: str, min_score_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    LLM 응답을 파싱하여 필터링 결과를 추출합니다.
    Parses LLM response to extract filtering results.
    
    Args:
        response_text: LLM의 응답 텍스트 / LLM response text
        min_score_threshold: (사용 안 함 - 하위 호환성 유지용) / (unused - for backward compatibility)
        
    Returns:
        List[Dict]: 필터링된 영화 목록 / Filtered movie list
                    [{"movieId": "...", "reason": "..."}]
    """
    try:
        # JSON 부분만 추출
        json_str = response_text.strip()
        
        # 마크다운 코드 블록 제거
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else lines[1]
        
        # JSON 파싱
        result = json.loads(json_str)
        
        # JSON 객체에서 movies 배열 추출
        if isinstance(result, dict):
            movies_array = result.get("movies", [])
        elif isinstance(result, list):
            # 하위 호환성: 배열도 허용
            movies_array = result
        else:
            return []
        
        if not isinstance(movies_array, list):
            return []
        
        # 각 항목 검증
        validated_results = []
        for item in movies_array:
            if not isinstance(item, dict):
                continue
            
            # 필수 필드 확인: movieId와 reason만 필요
            if "movieId" not in item:
                continue
            
            # reason 기본값 설정
            if "reason" not in item or not item["reason"]:
                item["reason"] = "사용자 요청에 부합하는 영화입니다."
            
            # movieId와 reason만 포함
            validated_item = {
                "movieId": item["movieId"],
                "reason": item["reason"]
            }
            
            validated_results.append(validated_item)
        
        return validated_results
        
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON Parsing Error: {str(e)}")
        return []
    except Exception as e:
        print(f"⚠️  Parsing Error: {str(e)}")
        return []


async def filter_movies_with_llm(
    user_query: str,
    movies: List[Dict[str, Any]],
    max_results: int = 10,
    min_score_threshold: float = 0.5  # 최소 점수 기준 / Minimum score threshold
) -> List[Dict[str, Any]]:
    """
    LLM을 사용하여 영화 목록을 필터링하고 우선순위를 매깁니다.
    Filters and prioritizes movie list using LLM.
    
    Args:
        user_query: 사용자의 자연어 쿼리 / User's natural language query
        movies: 백엔드에서 받은 영화 목록 / Movie list from backend
        max_results: 반환할 최대 결과 수 (기본값: 10) / Maximum number of results to return (default: 10)
        min_score_threshold: 최소 점수 기준 / Minimum score threshold
        
    Returns:
        List[Dict]: 필터링 및 정렬된 영화 목록 / Filtered and sorted movie list
        [
            {
                "movieId": "12345",
                "matchScore": 0.95,
                "reason": "우주를 배경으로 한 감동적인 스토리를..."
            }
        ]
        
    Raises:
        HTTPException: 처리 중 오류 발생 시
    """
    try:
        # 입력 검증
        if not user_query or not user_query.strip():
            raise HTTPException(status_code=400, detail="User Query is empty.")
        
        if not movies or len(movies) == 0:
            return []
        
        # 1. 영화 정보를 LLM 친화적 형식으로 변환
        movies_json = prepare_movies_for_llm(movies)
        
        # 2. 프롬프트 생성 (YAML 파일에서 로드)
        messages = await create_filter_prompt(user_query, movies_json)
        
        # 3. LLM 호출
        response = await llm.ainvoke(messages)
        response_text = response.content
        
        # 4. 응답 파싱 (최소 점수 기준 적용)
        filtered_results = parse_filter_response(response_text, min_score_threshold=min_score_threshold)
        
        # 5. 최대 결과 수 제한
        filtered_results = filtered_results[:max_results]
        
        return filtered_results
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Movie Filtering Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"영화 필터링 중 오류가 발생했습니다: {str(e)}"
        )


async def merge_filtered_results_with_movies(
    filtered_results: List[Dict[str, Any]],
    original_movies: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    필터링 결과를 원본 영화 정보와 병합합니다.
    Merges filtered results with original movie information.
    
    Args:
        filtered_results: LLM이 필터링한 결과 (movieId, reason) / 
                         LLM filtered results (movieId, reason)
        original_movies: 백엔드에서 받은 원본 영화 목록 / 
                        Original movie list from backend
        
    Returns:
        List[Dict]: 병합된 영화 목록 (원본 정보 + aiReason) / 
                   Merged movie list (original info + aiReason)
    """
    # movieId를 키로 하는 딕셔너리 생성
    movie_map = {str(movie.get("id")): movie for movie in original_movies}
    
    merged_results = []
    
    for filtered in filtered_results:
        movie_id = str(filtered.get("movieId"))
        
        if movie_id in movie_map:
            # 원본 영화 정보 복사
            merged_movie = movie_map[movie_id].copy()
            
            # LLM 결과 추가 (reason만)
            merged_movie["aiReason"] = filtered.get("reason", "")
            
            merged_results.append(merged_movie)
    
    return merged_results

