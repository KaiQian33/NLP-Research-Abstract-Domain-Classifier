## WOS-46985 Research Abstract Domain Classifier

## Repository Background and Attribution

This repository is forked from the original project maintained by [Avelyn-033/AI](https://github.com/Avelyn-033/AI).

The fork provides a separate workspace for deployment, testing, and further development while acknowledging the original project as the foundation of this repository.

## Project Files

### Exploratory Analysis and Modeling

**`exploratory_analysis_and_modeling.py`**

The full analysis script covers:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Decision Tree (DT) classification
- Logistic Regression (LR) classification
- Bidirectional Long Short-Term Memory (BiLSTM) classification
- Ablation study

The script is used to generate figures and tables for project documentation.

**Running the analysis script (optional):**

1. Install the required packages:

   ```bash
   pip install pandas numpy matplotlib scikit-learn nltk openpyxl wordcloud
   ```

2. For the BiLSTM section, install TensorFlow:

   ```bash
   pip install tensorflow-cpu
   ```

3. Open `exploratory_analysis_and_modeling.py` and update `DATA_PATH` (approximately line 30) to point to your local copy of `Data.xlsx`.

4. Run the script:

   ```bash
   python exploratory_analysis_and_modeling.py
   ```

### Streamlit Application

| File                     | Description                                                                                   |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| `app.py`                 | Streamlit application entry point.                                                            |
| `nlp_utils.py`           | Shared text-cleaning function.                                                                |
| `train.py`               | Trains Decision Tree and Logistic Regression models and saves them to the `models` directory. |
| `colab_lstm_training.py` | Trains the BiLSTM model in Google Colab and saves the resulting model files.                  |
| `requirements.txt`       | Python dependencies required by the application.                                              |
| `runtime.txt`            | Python version configuration for deployment, where supported by the hosting platform.         |
| `models/`                | Contains pre-trained models included in the repository.                                       |

## Running the Streamlit Application Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/KaiQian33/NLP.git
   cd NLP
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:

   ```bash
   streamlit run app.py
   ```

   Alternatively, if the `streamlit` command is not recognised:

   ```bash
   python -m streamlit run app.py
   ```

## Dataset

The dataset used in this project is sourced from Mendeley Data:

- **Source:** [Web of Science Dataset — Version 6](https://data.mendeley.com/datasets/9rw3vkcfy4/6)
- **DOI:** [10.17632/9rw3vkcfy4.6](https://doi.org/10.17632/9rw3vkcfy4.6)

Please refer to the original dataset page for its authorship, licence, and citation requirements.

## Acknowledgements

We acknowledge the original project contributors and the creators of the dataset for their respective contributions.
