#!/usr/bin/env python3
"""
MLflow 데이터베이스 초기화 스크립트
- MySQL 데이터베이스 존재 확인
- 데이터베이스가 없는 경우 생성
"""

import os
import pymysql
import sys

# 환경 변수에서 설정 가져오기
DB_HOST = os.environ.get('DB_HOST', 'mysql')
DB_PORT = int(os.environ.get('DB_PORT', 3306))
DB_USER = os.environ.get('DB_USER', 'mlflow_user')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'mlflow_password')
DB_NAME = os.environ.get('DB_NAME', 'mlflow')

def init_database():
    """MLflow 데이터베이스 초기화"""
    try:
        # MySQL 연결 (데이터베이스 없이)
        connection = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD
        )

        try:
            with connection.cursor() as cursor:
                # 데이터베이스 존재 확인
                cursor.execute(f"SHOW DATABASES LIKE '{DB_NAME}'")
                result = cursor.fetchone()
                
                # 데이터베이스가 없는 경우 생성
                if not result:
                    print(f"데이터베이스 '{DB_NAME}'가 존재하지 않습니다. 생성합니다...")
                    cursor.execute(f"CREATE DATABASE {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                    print(f"데이터베이스 '{DB_NAME}'가 생성되었습니다.")
                else:
                    print(f"데이터베이스 '{DB_NAME}'가 이미 존재합니다.")
                
                # 권한 부여
                cursor.execute(f"GRANT ALL PRIVILEGES ON {DB_NAME}.* TO '{DB_USER}'@'%'")
                cursor.execute("FLUSH PRIVILEGES")
                
        finally:
            connection.close()

    except pymysql.MySQLError as e:
        print(f"MySQL 오류 발생: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("MLflow 데이터베이스 초기화 시작...")
    init_database()
    print("MLflow 데이터베이스 초기화 완료.")