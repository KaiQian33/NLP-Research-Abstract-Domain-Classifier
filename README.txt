WOS-46985 Research Abstract Domain Classifier

FILES
------------------------------------
exploratory_analysis_and_modeling.py   
Full analysis script (EDA, preprocessing, DT/LR/BiLSTM, ablation study) — used for the Documentation figures/tables.

RUN exploratory_analysis_and_modeling.py (optional, for reference only)
------------------------------------
1. Install: pip install pandas numpy matplotlib scikit-learn nltk openpyxl wordcloud
   (BiLSTM section also needs: pip install tensorflow-cpu)
2. Open the script and update DATA_PATH (line ~30) to point to your
   local copy of Data.xlsx.

streamlit_app
------------------------------------- 
  app.py                               Streamlit app entry point.
  nlp_utils.py                         Shared text-cleaning function.
  train.py                             Trains DT + LR, saves to models.
  colab_lstm_training.py               Trains BiLSTM on Google Colab, saves to models.
  requirements.txt / runtime.txt       Dependencies + Python version for deployment.
  models                              Pre-trained models (already included).

LIVE DEMO
-------------------------------------
App:    https://ai-assignment-bmcs2074.streamlit.app/
GitHub: https://github.com/Avelyn-033/AI

RUN LOCALLY
-----------
pip install -r streamlit_app/requirements.txt
cd streamlit_app
streamlit run app.py
(if "streamlit" isn't recognised: python -m streamlit run app.py)