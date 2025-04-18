import pandas as pd
import mlflow
import os
import yaml
from datetime import datetime

def track_training_results(csv_path, yaml_path=None, experiment_name=None):
    """
    Track training results from CSV file and parameters from YAML file using MLflow.
    
    Args:
        csv_path (str): Path to the CSV file containing training results
        yaml_path (str, optional): Path to the YAML file containing parameters
        experiment_name (str, optional): Name of the MLflow experiment
    """
    # 1. CSV 파일 로드
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV file with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # 2. YAML 파일 로드 (파라미터)
    params = {}
    if yaml_path and os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                params = yaml.safe_load(f)
            print(f"Loaded parameters from YAML file: {yaml_path}")
        except Exception as e:
            print(f"Error loading YAML file: {e}")
    
    # 3. MLflow 실험 설정
    if experiment_name is None:
        experiment_name = f"hpo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    mlflow.set_experiment(experiment_name)
    
    # 4. 실험 시작
    with mlflow.start_run(run_name=f"training_run_{os.path.basename(csv_path)}"):
        # 학습 파라미터 기록
        mlflow.log_param("csv_source", os.path.basename(csv_path))
        mlflow.log_param("total_epochs", len(df))
        
        # YAML에서 로드한 모든 파라미터 기록
        if params:
            for param_name, param_value in params.items():
                # None, 리스트, 딕셔너리 등의 특수 타입 처리
                if param_value is None:
                    param_value = "None"
                elif isinstance(param_value, (list, dict)):
                    param_value = str(param_value)
                
                mlflow.log_param(param_name, param_value)
        
        # 5. 각 에포크별 데이터 기록
        for i, (_, row) in enumerate(df.iterrows()):
            epoch = int(row['epoch'])
            
            # 단계 시작
            with mlflow.start_run(run_name=f"epoch_{epoch}", nested=True):
                # 모든 컬럼을 메트릭으로 기록 (epoch 제외)
                for col in df.columns:
                    if col != 'epoch':
                        # MLflow에서 허용되지 않는 특수 문자(괄호 등)를 언더스코어로 대체
                        metric_name = col.replace("(", "_").replace(")", "_")
                        mlflow.log_metric(metric_name, float(row[col]))
                
                # epoch를 별도 태그로 기록
                mlflow.set_tag("epoch", epoch)
            
            # 6. 마지막 에포크의 메트릭을 부모 실행에도 기록 (실험 간 비교를 위해)
            if i == len(df) - 1:  # 마지막 에포크인 경우
                print(f"Logging final metrics (epoch {epoch}) to parent run")
                for col in df.columns:
                    if col != 'epoch':
                        metric_name = col.replace("(", "_").replace(")", "_")
                        # 접두어를 추가하여 부모 실행의 메트릭 구분
                        parent_metric_name = f"final_{metric_name}"
                        mlflow.log_metric(parent_metric_name, float(row[col]))
                
                # 최고 성능 메트릭도 기록 (mAP 등)
                best_map = df['metrics/mAP50-95_B_'].max()
                mlflow.log_metric("best_mAP50-95", best_map)
    
    print(f"Training results successfully tracked in MLflow experiment: {experiment_name}")


def argparse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Track training results using MLflow")
    parser.add_argument("--trial_name", type=str, help="Trial name (ex: trial_00068)")
    return parser.parse_args()

if __name__ == "__main__":
    
    args = argparse_args()
    
    csv_path = fr"c:\Users\steve\OneDrive\Desktop\hpo_results\{args.trial_name}\results.csv"
    yaml_path = fr"c:\Users\steve\OneDrive\Desktop\hpo_results\{args.trial_name}\args.yaml"
    experiment_name = "yolo_training_experiment"
    
    track_training_results(csv_path, yaml_path, experiment_name)