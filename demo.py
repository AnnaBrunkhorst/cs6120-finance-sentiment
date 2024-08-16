import streamlit as st
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from utils.GloveUtils import GloveUtils

BERT_UNCASED = ("google-bert/bert-base-uncased", "./models/old_bert_uncased")
BERT_CASED = ("google-bert/bert-base-cased", "./models/bert-cased")
DISTIL_BERT = ("distilbert/distilbert-base-uncased", "./models/distilbert")
FIN_BERT = ("ProsusAI/finbert", "./models/finbert")

GLOVE_NN = ("GloVe-NN", "./models/2048_final_ep_300")

label_map = [
    'Bearish',
    'Somewhat-Bearish',
    'Neutral',
    'Somewhat-Bullish',
    'Bullish'
]

condensed_label_map = [
    'Bearish',
    'Neutral',
    'Bullish'
]


@st.cache_data
def load_embeddings():
    GLOVE_PATH = "./data/glove.6B.100d.txt"
    glove_utils = GloveUtils(GLOVE_PATH, max_dims=64)
    glove_utils.create_glove_emb_layer(trainable=True)
    return glove_utils


glove_utils = load_embeddings()


def run_nn_model(text, model_name, path):
    # Send to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(path)
    model.to(device)

    text = text.split(" ")[:64]
    glove_ids = glove_utils.get_embedding_indices([text])
    glove_ids = glove_ids.to(device)
    # Perform inference
    with torch.no_grad():
        probs = model(glove_ids)

    pred = int(torch.argmax(probs, dim=-1).cpu().numpy()[0])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return label_map[pred]


def run_transformer_model(text, model_name, path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(path)

    # Send to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # Apply softmax and get the predicted class
    probs = F.softmax(logits, dim=-1)
    pred = int(torch.argmax(probs, dim=-1).cpu().numpy()[0])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if "fin" in model_name:
        return condensed_label_map[pred]
    else:
        return label_map[pred]


# Title of the app
st.title("Finance News Sentiment Analysis")

# Text input for user to enter text
user_input = st.text_area("Enter a news headline")

# Dropdown for model selection (assuming you have multiple models)
model_option = st.selectbox(
    "Choose a model:",
    (BERT_CASED[0], BERT_UNCASED[0], DISTIL_BERT[0], FIN_BERT[0], GLOVE_NN[0])
)

# Button to trigger prediction
if st.button("Analyze Sentiment"):
    if model_option == GLOVE_NN[0]:
        sentiment = run_nn_model(user_input, *GLOVE_NN)
    elif model_option == BERT_CASED[0]:
        sentiment = run_transformer_model(user_input, *BERT_CASED)
    elif model_option == BERT_UNCASED[0]:
        sentiment = run_transformer_model(user_input, *BERT_UNCASED)
    elif model_option == FIN_BERT[0]:
        sentiment = run_transformer_model(user_input, *FIN_BERT)
    else:
        sentiment = run_transformer_model(user_input, *DISTIL_BERT)

    # Display the prediction
    st.write(f"Prediction: {sentiment}")
    st.write(f"Using model: {model_option}")
