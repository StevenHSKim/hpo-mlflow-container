# MLflow 시각화 컨테이너

이 프로젝트는 MLOps 파이프라인을 위한 MLflow 시각화 컨테이너를 구현합니다. HPO(Hyperparameter Optimization) 실험 결과를 추적하고 시각화하는 데 사용됩니다.

## 구성 요소

이 컨테이너 설정은 다음 서비스로 구성됩니다:

1. **MLflow 서버**: 실험 추적 및 시각화 UI 제공
2. **MySQL 데이터베이스**: 실험 메타데이터 저장
3. **MinIO 객체 스토리지**: 모델 아티팩트 저장 (모델 파일, 로그 등)
4. **MinIO 초기화 서비스**: 필요한 버킷 생성 및 설정

## 시스템 요구사항

- Docker 및 Docker Compose가 설치되어 있어야 합니다.
- 최소 4GB RAM (권장: 8GB 이상)
- 최소 10GB 디스크 공간

## 설치 및 실행 방법

1. 저장소 클론:
   ```bash
   git clone <repository-url>
   cd mlflow-container
   ```

2. 환경 변수 설정 (필요 시 config.env 파일 수정):
   ```bash
   # 기본값 사용 또는 필요에 따라 config.env 파일 편집
   ```

3. 컨테이너 시작:
   ```bash
   docker-compose up -d
   ```

4. 시각화 UI 접근:
   - MLflow UI: http://localhost:5000
   - MinIO 콘솔: http://localhost:9001 (접속 정보: minio_access_key / minio_secret_key)

## MLflow API 사용 예시

### Python 클라이언트에서 실험 로깅

```python
import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# MLflow 서버 연결 설정
mlflow.set_tracking_uri("http://localhost:5000")

# 실험 생성 또는 사용
experiment_name = "my_hpo_experiment"
mlflow.set_experiment(experiment_name)

# 학습 데이터 (예시)
X, y = np.random.rand(100, 10), np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MLflow 실험 시작
with mlflow.start_run(run_name="trial_001"):
    # 하이퍼파라미터 기록
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2
    }
    mlflow.log_params(params)
    
    # 모델 학습
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # 성능 지표 기록
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    
    # 모델 아티팩트 저장
    mlflow.sklearn.log_model(model, "model")
    
    # 추가 아티팩트 (예: 설정 파일) 저장
    with open("config.yaml", "w") as f:
        f.write("model_type: RandomForest\nversion: 1.0")
    mlflow.log_artifact("config.yaml")
```

## HPO 파트와 연동

이 MLflow 컨테이너는 HPO 실행 결과를 추적하고 시각화하는 데 사용됩니다. 

1. HPO 파트에서는 MLflow 클라이언트를 사용하여 각 trial의 결과를 기록합니다.
2. 학습된 모델, 설정 파일, 로그 등은 MinIO 스토리지에 저장됩니다.
3. 실험 메타데이터는 MySQL 데이터베이스에 저장됩니다.
4. 사용자는 MLflow UI를 통해 실험 결과를 시각적으로 분석할 수 있습니다.

## 트러블슈팅

- **MLflow UI 접속 오류**: 컨테이너 로그를 확인하고 MLflow 서버가 정상적으로 실행되고 있는지 확인합니다.
  ```bash
  docker logs mlflow_server
  ```

- **MinIO 연결 오류**: MinIO 서버가 실행 중인지 확인하고 접속 정보가 올바른지 확인합니다.
  ```bash
  docker logs mlflow_minio
  ```

- **데이터베이스 연결 오류**: MySQL 서버가 실행 중인지 확인하고 접속 정보가 올바른지 확인합니다.
  ```bash
  docker logs mlflow_mysql
  ```