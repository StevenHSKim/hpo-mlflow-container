FROM python:3.9-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-libmysqlclient-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mlflow

# 필요한 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 스크립트 복사 및 권한 설정
COPY scripts/ /mlflow/scripts/
RUN chmod +x /mlflow/scripts/entrypoint.sh

# MLflow가 사용할 포트 노출
EXPOSE 5000

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1

# 진입점 설정
ENTRYPOINT ["/mlflow/scripts/entrypoint.sh"]