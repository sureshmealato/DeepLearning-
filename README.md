# DeepLearning-

A structured Deep Learning project using TensorFlow & Keras.

## Tech Stack
- Python 3.10+
- TensorFlow (>=2.12)
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- Jupyter

## Project Structure
```
DeepLearning-/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── saved_models/
```

## Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / Mac
   venv\Scripts\activate      # Windows
   ```
2. Upgrade pip:
   ```bash
   pip install --upgrade pip
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Phase Roadmap
- **Phase 1:** Project setup & environment (TensorFlow)
- **Phase 2:** Data ingestion and preprocessing
- **Phase 3:** Model training and evaluation
- **Phase 4:** Experiment tracking and iteration
- **Phase 5:** Deployment and inference

## Data Pipeline (Phase 2)
- `src/data/data_loader.py` loads MNIST, builds train/validation/test splits, and constructs `tf.data` pipelines.
- `src/utils/config.py` centralizes batch size, image shape, and reproducibility settings.
- `notebooks/01_data_exploration.ipynb` explores the dataset, visualizes samples, and validates preprocessing.

## Training & Evaluation (Phase 3)
- `src/models/ann.py` defines a baseline ANN model with a compiled TensorFlow graph.
- `src/training/trainer.py` trains models with early stopping and checkpoints.
- `src/evaluation/metrics.py` reports loss/accuracy plus confusion matrix and classification report.
- `notebooks/02_train_ann.ipynb` runs the end-to-end workflow and visualizes metrics.

## Experiment Tracking & Comparison (Phase 4)
- `src/models/cnn.py` adds a baseline CNN model for spatial feature learning.
- `src/models/__init__.py` exposes a model registry and name-based factory for experiment selection.
- `experiments/` stores YAML configs for reproducible runs.
- `src/training/trainer.py` logs TensorBoard metrics to `logs/<experiment>/`.
- `src/evaluation/metrics.py` can evaluate multiple models and persist JSON reports in `reports/`.
- `notebooks/03_experiment_comparison.ipynb` compares ANN vs CNN histories and metrics.

## Validation Check
Run the following to confirm TensorFlow is available:
```python
import tensorflow as tf
print(tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))
```
