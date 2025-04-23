#!/usr/bin/env bash
set -e   # 에러 발생 시 스크립트 중단

PYTHON="python"
SCRIPT="c:/Users/steve/hpo-mlflow-container/hpo_mlflow.py"

echo "=== Running grid_trial_00068 ==="
$PYTHON "$SCRIPT" --folder_name grid_trial_00068

echo "=== Running grid_trial_00078 ==="
$PYTHON "$SCRIPT" --folder_name grid_trial_00078

echo "=== Running grid_trial_00087 ==="
$PYTHON "$SCRIPT" --folder_name grid_trial_00087

echo "=== Running grid_trial_00088 ==="
$PYTHON "$SCRIPT" --folder_name grid_trial_00088

echo "=== Running grid_trial_00092 ==="
$PYTHON "$SCRIPT" --folder_name grid_trial_00092

echo "🎉 All grid_trials finished."
