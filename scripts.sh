#!/usr/bin/env bash
set -e   # 에러 발생 시 스크립트 중단

PYTHON="python"
SCRIPT="c:/Users/steve/hpo-mlflow-container/hpo_mlflow.py"

echo "=== Running grid_trial_00000 ==="
$PYTHON "$SCRIPT" --folder_name grid_trial_00000

echo "=== Running grid_trial_00001 ==="
$PYTHON "$SCRIPT" --folder_name grid_trial_00001

echo "=== Running grid_trial_00002 ==="
$PYTHON "$SCRIPT" --folder_name grid_trial_00002

echo "=== Running grid_trial_00003 ==="
$PYTHON "$SCRIPT" --folder_name grid_trial_00003

echo "=== Running grid_trial_00004 ==="
$PYTHON "$SCRIPT" --folder_name grid_trial_00004

echo "=== Running grid_trial_00005 ==="
$PYTHON "$SCRIPT" --folder_name grid_trial_00005

echo "=== Running grid_trial_00006 ==="
$PYTHON "$SCRIPT" --folder_name grid_trial_00006

echo "=== Running grid_trial_00007 ==="
$PYTHON "$SCRIPT" --folder_name grid_trial_00007

echo "=== Running grid_trial_00008 ==="
$PYTHON "$SCRIPT" --folder_name grid_trial_00008

echo "=== Running grid_trial_00009 ==="
$PYTHON "$SCRIPT" --folder_name grid_trial_00009

echo "🎉 All grid_trials finished."
