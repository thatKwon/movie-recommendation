import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# .env 파일 로드 (있는 경우)
load_dotenv()

def get_secret_from_aws(secret_name="prod/AppBeta/apikey", region_name="ap-northeast-2"):
    """
    AWS Secrets Manager에서 시크릿 값을 가져옵니다. (선택적)
    Retrieves secret values from AWS Secrets Manager. (Optional)
    """
    try:
        import boto3
        import json
        
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )
        
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        
        secret = get_secret_value_response['SecretString']
        return json.loads(secret)
    except ImportError:
        # boto3가 설치되지 않은 경우
        return None
    except Exception:
        # AWS 연결 실패 또는 권한 없음
        return None

def initialize_environment():
    """
    환경 변수 검증 및 설정 (우선순위: 환경변수 > AWS Secrets Manager)
    Validates and configures environment variables (Priority: env vars > AWS Secrets Manager)
    """
    
    # 1. 먼저 환경변수에서 API 키 확인
    openai_key = os.getenv("OPENAI_API_KEY")
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    
    # 2. 환경변수에 없으면 AWS Secrets Manager 시도
    if not openai_key or not langsmith_key:
        secrets = get_secret_from_aws()
        if secrets:
            openai_key = openai_key or secrets.get("OpenAI")
            langsmith_key = langsmith_key or secrets.get("Langsmith")
    
    # 3. 필수 API 키 검증
    required_env_vars = {
        "OPENAI_API_KEY": openai_key,
        "LANGSMITH_API_KEY": langsmith_key
    }

    missing_vars = [var for var, value in required_env_vars.items() if not value]
    if missing_vars:
        raise EnvironmentError(
            f"❌ 필수 환경 변수가 누락되었습니다: {', '.join(missing_vars)}\n\n"
            "✅ 다음 중 하나의 방법으로 API 키를 설정해주세요:\n\n"
            "방법 1) .env 파일 생성 (ai-service/.env):\n"
            "   OPENAI_API_KEY=sk-your-openai-key\n"
            "   LANGSMITH_API_KEY=lsv2_pt_your-langsmith-key\n\n"
            "방법 2) 환경변수 직접 설정:\n"
            "   export OPENAI_API_KEY=sk-your-openai-key\n"
            "   export LANGSMITH_API_KEY=lsv2_pt_your-langsmith-key\n\n"
            "방법 3) AWS Secrets Manager 사용 (선택적, boto3 설치 필요)"
        )

    # 4. Langsmith 환경 변수 설정
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_API_KEY"] = required_env_vars["LANGSMITH_API_KEY"]
    os.environ["LANGSMITH_PROJECT"] = "movie-recommendation"
    os.environ["OPENAI_API_KEY"] = required_env_vars["OPENAI_API_KEY"]
    
    print("✅ API 키 로드 성공!")
    print(f"   - OpenAI: {openai_key[:10]}...")
    print(f"   - Langsmith: {langsmith_key[:15]}...")
    
    return required_env_vars

# 환경 변수 초기화
env_vars = initialize_environment()

# 모델 선택 (.env에서 설정 가능)
# 속도 우선: gpt-4o-mini (1-3초) ⚡⚡⚡
# 품질 우선: gpt-4o (3-5초) ⚡⚡
# 레거시: gpt-4 (10-15초) 🐌
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

llm = ChatOpenAI(
    model=model_name,
    temperature=0.0,
    model_kwargs={
        "response_format": {"type": "json_object"}  # JSON 모드 강제
    }
)

print(f"🤖 LLM 모델: {model_name}")

# 호환성을 위한 별칭
accurate_llm = llm
