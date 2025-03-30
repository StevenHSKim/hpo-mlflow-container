#!/bin/bash
set -e

# 환경 변수 로드
source /mlflow/config.env

# MySQL 연결 확인
echo "MySQL 서버 연결 대기 중..."
max_retries=30
attempt=0

while [ $attempt -lt $max_retries ]; do
    attempt=$((attempt+1))
    echo "시도 $attempt/$max_retries..."
    
    if python -c "import pymysql; pymysql.connect(host='$DB_HOST', user='$DB_USER', password='$DB_PASSWORD', port=$DB_PORT)" 2>/dev/null; then
        echo "MySQL 서버와 연결되었습니다!"
        break
    fi
    
    if [ $attempt -eq $max_retries ]; then
        echo "MySQL 서버에 연결할 수 없습니다. 30번 시도 후 실패."
        exit 1
    fi
    
    echo "MySQL 서버에 연결할 수 없습니다. 5초 후 재시도합니다..."
    sleep 5
done

# DB 초기화 스크립트 실행 (필요시)
echo "MLflow 데이터베이스 초기화 중..."
python /mlflow/scripts/initialize_db.py

# MinIO 연결 설정 (S3 호환 스토리지)
echo "MinIO 아티팩트 스토어 연결 설정 중..."
export AWS_ACCESS_KEY_ID=$MINIO_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=$MINIO_SECRET_KEY
export MLFLOW_S3_ENDPOINT_URL="http://${MINIO_HOST}:${MINIO_PORT}"

# MLflow 서버 시작
echo "MLflow 서버 시작 중..."
exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "mysql+pymysql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}" \
    --default-artifact-root "s3://${MINIO_BUCKET}" \
    --serve-artifacts