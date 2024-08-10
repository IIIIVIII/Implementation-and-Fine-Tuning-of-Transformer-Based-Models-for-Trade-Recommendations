# Blockhouse Machine Learning Engineer Work Trial

## Overview

This project is part of a work trial for the Machine Learning Engineer position at Blockhouse. The primary task was to implement and fine-tune a transformer-based model to generate trade recommendations based on market data. The project involved creating a hybrid model using both LSTM and Transformer layers, followed by extensive fine-tuning and evaluation to optimize the model's performance. Additionally, backtesting was performed to assess the model's real-world applicability.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data](#data)
- [Model Implementation](#model-implementation)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The project directory is organized as follows:

```
blockhouse/
├── bilstm_model_evaluation.py               # Initial evaluation of the BiLSTM model
├── data_preparation.py                      # Data preprocessing and feature engineering
├── model_evaluation.py                      # Modular evaluation framework
├── model_evaluation_with_backtest.py        # Model evaluation including backtesting
├── model_implementation.py                  # Consolidated model implementation
├── model_implementation_lstm.py             # Basic LSTM model implementation
├── model_implementation_with_bidirectional_lstm.py # BiLSTM model implementation
├── model_implementation_with_features.py    # Model implementation with additional features
├── model_implementation_with_more_features.py # Further feature enhancement
├── report_generation.py                     # Automated report generation
├── test_environment.py                      # Environment setup validation
├── xnas-itch-20230703.tbbo.csv              # Market data used for training and evaluation
├── best_hybrid_model.pth                    # Best performing model weights
└── blockhouse-env/                          # Virtual environment directory (not included in GitHub repo)
```

## Installation

### Prerequisites

Ensure you have Python 3.7+ installed. You'll also need to install the following Python packages:

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `imblearn`

### Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/blockhouse-ml-engineer-trial.git
   cd blockhouse-ml-engineer-trial
   ```

2. **Create a Virtual Environment:**

   ```bash
   python3 -m venv blockhouse-env
   source blockhouse-env/bin/activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Activate the Environment:**

   Every time you work on the project, ensure the virtual environment is activated:

   ```bash
   source blockhouse-env/bin/activate
   ```

## Data

The market data used in this project is stored in the file `xnas-itch-20230703.tbbo.csv`. This dataset is critical for training and evaluating the models. It includes trade and market data that has been processed to include various technical indicators like RSI, MACD, and Bollinger Bands.

## Model Implementation

### 1. LSTM Model

The LSTM model is implemented in `model_implementation_lstm.py`. This script defines the architecture of the LSTM model, trains it on the market data, and saves the trained model.

### 2. BiLSTM Model

The BiLSTM model, which uses a bidirectional LSTM for better context understanding, is implemented in `model_implementation_with_bidirectional_lstm.py`.

### 3. Hybrid Model

The core of this project is the Hybrid model that combines LSTM and Transformer layers to leverage both sequential and attention-based modeling techniques. The implementation is split across several scripts:
- `model_implementation_with_features.py`
- `model_implementation_with_more_features.py`

### 4. Consolidated Implementation

To streamline and optimize the approach, the final model was implemented in `model_implementation.py`, integrating the best practices from previous iterations.

## Evaluation

### 1. Model Evaluation

Model evaluation is conducted using the `model_evaluation.py` and `model_evaluation_with_backtest.py` scripts. These scripts load the trained models and run them against the test set to generate accuracy, precision, recall, F1 scores, confusion matrices, and financial metrics through backtesting.

### 2. Report Generation

The `report_generation.py` script is responsible for generating visualizations and detailed reports of the model's performance. This includes plotting confusion matrices and analyzing the model's predictions.

## Results

After extensive training and evaluation, the final model achieved the following results:

- **Accuracy:** 0.94
- **Precision:** 0.94 (weighted avg)
- **Recall:** 0.94 (weighted avg)
- **F1 Score:** 0.94 (weighted avg)
- **Final Balance after Backtest:** $12,881.74

The confusion matrix and classification report generated during the evaluation phase are included in the `report_generation.py` script.

## Usage

To run the model training and evaluation, follow these steps:

1. **Run Data Preparation:**

   ```bash
   python data_preparation.py
   ```

2. **Train the Model:**

   Depending on the model you wish to train, run one of the following:

   ```bash
   python model_implementation_lstm.py
   ```

   or

   ```bash
   python model_implementation_with_bidirectional_lstm.py
   ```

3. **Evaluate the Model:**

   ```bash
   python model_evaluation_with_backtest.py
   ```

4. **Generate the Report:**

   ```bash
   python report_generation.py
   ```

## Future Work

- **Integration with PPO:** Explore the integration of the transformer model with a PPO implementation to leverage reinforcement learning in trading strategies.
- **Hyperparameter Tuning:** Further fine-tune the model's hyperparameters using grid search or Bayesian optimization.
- **Feature Engineering:** Experiment with additional technical indicators and market features to enhance model performance.

## Contributing

Contributions are welcome! If you have ideas for improvement or new features, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

