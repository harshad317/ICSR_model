import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME3 = "DKTech/ICSR_classification_model_biolinkbert_2"
tokenizer3 = AutoTokenizer.from_pretrained(MODEL_NAME3, use_auth_token= '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')
model3 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME3, use_auth_token= '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')


def predict_sentiment3(text):
    inputs = tokenizer3(text, return_tensors="pt", truncation=True, padding=True, max_length= 512)
    outputs = model3(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    sentiment3 = torch.argmax(probabilities).item()
    return sentiment3, probabilities.tolist()[0]

st.title("ICSR classifier")
title_input = st.text_input("Enter your title here:")
text_input = st.text_area("Enter your text here:")
text = title_input + " " + text_input

if st.button("Analyze"):
    #sentiment = predict_sentiment(text)
    pred3, proba3 = predict_sentiment3(text)

    sentiment, probs = predict_sentiment3(text)
    st.header('Bio_link_2 model result')
    if sentiment == 1:
        st.success("ICSR")
    else:
        st.error("Discarded NICSR")
    
    st.write(f"Probability of Discarded ICSR {probs[0]:.2f}")
    st.write(f"Probability of ICSR {probs[1]:.2f}")

    preds = []
    preds.append(pred3)
    st.header("Final result")
    if sum(preds) > len(preds) / 2:
        sentiment =  1
    else:
        sentiment = 0
    
    if sentiment == 1:
        st.success("ICSR")
    elif sentiment == 0:
        st.error(f"Discarded ICSR" )
    else:
        st.error(f"Not sure")
