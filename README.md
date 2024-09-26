# CS6120: Sentiment Analysis in Finance by TEAM103

### Team:
Ashish Magadum
Anna Brunkhorst
Nader Lobandi

### Dependencies
To run the setup, the following packages need to installed:

- **Python 3.12+**, Jupyter Notebook
- NLP/data: NLTK, SpaCy, Gensim, Pandas
- ML: NumPy, PyTorch, Transformers, Datasets, evaluate
- GPU-acceleration: CUDA-toolkit
- HuggingFace: Huggingface-hub, Transformers, Datasets, Evaluate, Accelerate
- UI: StreamLit
- Plots: Matplotlib, Seaborn

### Project Navigation

````
/                                           # Project root
    /data                                   # All datasets are present here
    /models                                 # All saved models go here
    /utils 
        DataUtils.py                        # Data utility functions 
        GloveUtils.py                       # Glove embedding utility functions 
    alpha_vantage_bert.ipynb                # Fine-tuning of BERT-cased, uncased and DistilBERT 
    alpha_vantage_finbert.ipynb             # Fine-tuning of FinBERT 
    alpha_vantage_glove.ipynb               # Training of GloVE+NN model 
    alpha_vantage_log_reg.ipynb             # Training of Logistic regression model 
    demo.py                                 # Entry point to the app. streamlit run demo.py                             
    EDA_data.py                             # Exploratory Data Analysis 
    env                                     # File to place the Vantage Free API key 
    evaluate_models.ipynb                   # Evaluate transformer and neural network models 
    train_test_prep.ipynb                   # train test split preparation 
    unit_tests.py                           # unit tests
    yahoo_bert.ipynb                        # Fine-tuning of BERT model on Yahoo dataset 
    yahoo_glove.ipynb                       # Training of GloVE+NN model on Yahoo 
    yahoo_log_reg.ipynb                     # Training of Logistic Reg on Yahoo 
    README.md                               # This file
````

### Instructions to run
- Ensure that all the dependencies are installed as mentioned above.
- Ensure all models are placed under `./models`. 
- Run the following command in the project root to execute the streamlit application. 
`streamlit run demo.py`
