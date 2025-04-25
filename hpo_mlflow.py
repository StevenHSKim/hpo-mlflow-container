import pandas as pd
import mlflow
import os
import yaml
from datetime import datetime

def set_mlflow_tracking_uri():
    """
    Set the MLflow tracking URI to a mlruns directory.
    """
    mlflow.set_tracking_uri("./mlruns")  ## TODO ##
    print("MLflow tracking URI set to mlruns directory.")
    

def track_training_results(args, csv_path, yaml_path=None, experiment_name=None, search_method=None, trial_name=None):
    """
    Track training results from CSV file and parameters from YAML file using MLflow.
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
    model_name = None
    model_yaml_path = None
    model_performance_metrics = {}
    dataset_tag = None
    model_scale_params = {}
    
    if yaml_path and os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                params = yaml.safe_load(f)
            print(f"Loaded parameters from YAML file: {yaml_path}")
            
            # 모델 파일 경로에서 모델명 추출
            if 'model' in params and params['model'] and isinstance(params['model'], str) and '.yaml' in params['model']:
                model_path = params['model']
                # 파일명만 추출 (경로 제거)
                model_file = os.path.basename(model_path)
                # .yaml 확장자 제거하고 모델명 추출
                model_name = model_file.split('.yaml')[0]
                print(f"Extracted model name: {model_name}")
                
                # 모델 YAML 파일 경로 구성
                folder_path = os.path.dirname(yaml_path)  # grid_trial_00000
                base_folder = os.path.basename(folder_path)  # grid_trial_00000
                model_yaml_folder = f"{base_folder}_model_yaml"  # grid_trial_00000_model_yaml
                parent_folder = os.path.dirname(folder_path)  # 부모 폴더
                model_yaml_path = os.path.join(parent_folder, model_yaml_folder, f"{model_name}.yaml")
                
                # 모델 YAML 파일이 존재하는지 확인하고 정보 추출
                if os.path.exists(model_yaml_path):
                    try:
                        with open(model_yaml_path, 'r') as f:
                            model_yaml = yaml.safe_load(f)
                            
                        # FPS, Latency, Parameters 추출
                        for metric in ['FPS', 'Latency', 'Parameters']:
                            if metric in model_yaml:
                                model_performance_metrics[metric] = model_yaml[metric]
                                print(f"Extracted {metric}: {model_performance_metrics[metric]}")
                        
                        # nc 값에 따라 dataset 태그 설정
                        if 'nc' in model_yaml:
                            nc_value = model_yaml['nc']
                            if nc_value == 80:
                                dataset_tag = "COCO"
                                print(f"Set dataset tag to COCO based on nc={nc_value}")
                            elif nc_value == 20:
                                dataset_tag = "PascalVOC"
                                print(f"Set dataset tag to PascalVOC based on nc={nc_value}")
                            else:
                                print(f"Found nc={nc_value}, but no matching dataset tag defined")
                        
                        # scales.n의 두 번째와 세 번째 값 추출 (width와 max_channel)
                        if 'scales' in model_yaml and 'n' in model_yaml['scales']:
                            n_scale = model_yaml['scales']['n']
                            if isinstance(n_scale, list) and len(n_scale) >= 3:
                                # 두 번째 값을 width로 추출
                                model_scale_params['width'] = n_scale[1]
                                print(f"Extracted width parameter: {model_scale_params['width']}")
                                
                                # 세 번째 값을 max_channel로 추출
                                model_scale_params['max_channel'] = n_scale[2]
                                print(f"Extracted max_channel parameter: {model_scale_params['max_channel']}")
                            else:
                                print(f"scales.n format is unexpected: {n_scale}")
                        
                    except Exception as e:
                        print(f"Error loading model YAML file: {e}")
            
        except Exception as e:
            print(f"Error loading YAML file: {e}")
    
    mlflow.set_experiment(experiment_name)
    
    # 4. 실험 시작
    with mlflow.start_run(run_name=f"training_run_{trial_name}"):
        # 서치 방법론 태그 추가
        mlflow.set_tag("search_method", search_method)
        
        # 모델명이 추출되었으면 태그로 설정
        if model_name:
            mlflow.set_tag("model", model_name)
        
        # 데이터셋 태그가 있으면 추가
        if dataset_tag:
            mlflow.set_tag("dataset", dataset_tag)
        
        # 학습 파라미터 기록
        mlflow.log_param("csv_source", os.path.basename(csv_path))
        mlflow.log_param("total_epochs", len(df))
        
        # YAML에서 로드한 모든 파라미터 기록 (model 제외)
        if params:
            for param_name, param_value in params.items():
                # model 파라미터는 건너뛰고 태그로 처리
                if param_name == 'model' and model_name:
                    continue
                    
                # None, 리스트, 딕셔너리 등의 특수 타입 처리
                if param_value is None:
                    param_value = "None"
                elif isinstance(param_value, (list, dict)):
                    param_value = str(param_value)
                
                mlflow.log_param(param_name, param_value)
        
        # 모델 스케일 파라미터 기록 (width, max_channel)
        for param_name, param_value in model_scale_params.items():
            mlflow.log_param(param_name, param_value)
        
        # 모델 성능 메트릭 기록 (FPS, Latency, Parameters는 한 trial 안에서 동일하므로 step=0)
        for metric_name, metric_value in model_performance_metrics.items():
            mlflow.log_metric(metric_name, float(metric_value), step=0)
        
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
                
                # 중첩 실행에도 모델명 태그 추가
                if model_name:
                    mlflow.set_tag("model", model_name)
                
                # 중첩 실행에도 데이터셋 태그 추가
                if dataset_tag:
                    mlflow.set_tag("dataset", dataset_tag)
                
                # epoch를 파라미터로 기록
                mlflow.log_param("epoch", epoch)
                
                # 모델 성능 메트릭도 중첩 실행에 기록
                for metric_name, metric_value in model_performance_metrics.items():
                    mlflow.log_metric(metric_name, float(metric_value), step=0)
                
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
    
    # MLflow tracking URI 설정
    set_mlflow_tracking_uri()
    
    # 경로 설정
    base_dir = "c:/Users/steve/OneDrive/Desktop/hpo_results_new"  ## TODO ##
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