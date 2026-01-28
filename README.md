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

## Validation Check
Run the following to confirm TensorFlow is available:
```python
import tensorflow as tf
print(tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))
```
