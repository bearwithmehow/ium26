import mlflow
from mlflow.tracking import MlflowClient
import tempfile

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# find latest run in experiment
experiment = client.get_experiment_by_name("letterboxd-rating-mlp")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time DESC"], 
    max_results=1
)
run = runs[0]
run_id = run.info.run_id
print(f"Registering run: {run_id}")

model_uri = f"runs:/{run_id}/letterboxd_mlp"
mlflow.register_model(model_uri, name="letterboxd-rating-mlp")

# model card with metadata from the run
model_card = f"""
# Model Card: Letterboxd Rating MLP

## Description
A MLP network for predicting the average rating of a movie on Letterboxd based on numerical features.

## Architecture
- Input: {run.data.params.get('input_dim', '?')} features
- Hidden layers: 128 → 64 (ReLU, Dropout 0.2)
- Output: 1 (regression)

## Hyperparameters
- Optimizer: {run.data.params.get('optimizer', 'adam')}
- Learning rate: {run.data.params.get('lr', '?')}
- Epochs: {run.data.params.get('epochs', '?')}
- Batch size: {run.data.params.get('batch_size', '?')}
- Loss: {run.data.params.get('loss', 'mse')}

## Metrics (last epoch)
- Train MSE: {run.data.metrics.get('train_mse', '?'):.4f}
- Dev MSE:   {run.data.metrics.get('dev_mse', '?'):.4f}
- Dev RMSE:  {run.data.metrics.get('dev_rmse', '?'):.4f}

## Data
- Dataset: Letterboxd 10k Movies (Kaggle)
- Split: 60% train / 20% dev / 20% test

## Limitations
- Model does not account for textual features (title, description, cast)
- Data from a single platform — may not generalize to other platforms
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
    f.write(model_card)
    temp_path = f.name

with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact(temp_path, artifact_path="documentation")
    print("Model card zalogowana")

print(f"Model registered with name 'letterboxd-rating-mlp' and URI: {model_uri}")
print("FIN")