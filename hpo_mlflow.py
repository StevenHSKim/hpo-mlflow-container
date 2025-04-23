import pandas as pd
import mlflow
import os
import yaml
from datetime import datetime


def track_training_results(args, csv_path, yaml_path=None, experiment_name=None, search_method=None, trial_name=None):
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
    
    mlflow.set_experiment(experiment_name)
    
    # 4. 실험 시작
    with mlflow.start_run(run_name=f"training_run_{trial_name}"):
        # 서치 방법론 태그 추가
        mlflow.set_tag("search_method", search_method)
        
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
        
        # 5. 먼저 부모 실행에 모든 에포크의 메트릭을 기록 (추이를 볼 수 있도록)
        for _, row in df.iterrows():
            epoch = int(row['epoch'])
            
            # 모든 컬럼을 메트릭으로 기록 (epoch 제외)
            for col in df.columns:
                if col != 'epoch':
                    # MLflow에서 허용되지 않는 특수 문자(괄호 등)를 언더스코어로 대체
                    metric_name = col.replace("(", "_").replace(")", "_")
                    mlflow.log_metric(metric_name, float(row[col]), step=epoch)
            
        
        # 6. 각 에포크별 데이터를 중첩 실행에 기록
        for i, (_, row) in enumerate(df.iterrows()):
            epoch = int(row['epoch'])
            
            # 단계 시작
            with mlflow.start_run(run_name=f"epoch_{epoch}", nested=True):
                # 중첩 실행에도 서치 방법론 태그 추가
                if search_method:
                    mlflow.set_tag("search_method", search_method)
                
                # epoch를 파라미터로 기록
                mlflow.log_param("epoch", epoch)
                
                # 모든 컬럼을 메트릭으로 기록 (epoch 제외)
                for col in df.columns:
                    if col != 'epoch':
                        # MLflow에서 허용되지 않는 특수 문자(괄호 등)를 언더스코어로 대체
                        metric_name = col.replace("(", "_").replace(")", "_")
                        mlflow.log_metric(metric_name, float(row[col]), step=epoch)
    
    print(f"Training results successfully tracked in MLflow experiment: {experiment_name}")


def argparse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Track training results using MLflow")
    parser.add_argument("--folder_name", type=str, required=True, help="Folder name (ex: grid_trial_00092, tpe_trial_00068)")
    return parser.parse_args()

if __name__ == "__main__":
    args = argparse_args()
    
    # 경로 설정
    base_dir = r"c:\Users\steve\OneDrive\Desktop\hpo_results"
    folder_path = os.path.join(base_dir, args.folder_name)
    
    # 폴더명에서 search_method와 trial_name 추출
    folder_name = args.folder_name
    parts = folder_name.split("_trial_")
    search_method = parts[0]
    trial_name = f"trial_{parts[1]}"
    
    csv_path = os.path.join(folder_path, "results.csv")
    yaml_path = os.path.join(folder_path, "args.yaml")
    
    # 기본 experiment_name 설정
    experiment_name = f"{search_method}_yolo_training_experiment"
    
    track_training_results(args, csv_path, yaml_path, experiment_name, search_method, trial_name)