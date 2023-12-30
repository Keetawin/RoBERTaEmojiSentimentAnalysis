
import torch
import gradio as gr
from model import SentimentAnalysisModel
from timeit import default_timer as timer

# Load the pre-trained sentiment analysis model
model = SentimentAnalysisModel(bert_model_name="SamLowe/roberta-base-go_emotions", num_labels=7)
model.load_state_dict(torch.load("best_model_75.pth", map_location=torch.device('cpu')), strict=False)

model.eval()

# Mapping from predicted class to emoji
emoji_to_emotion = {
    0: 'joy 😆',
    1: 'fear 😱',
    2: 'anger 😡',
    3: 'sadness 😭',
    4: 'disgust 🤮',
    5: 'shame 😳',
    6: 'guilt 😞'
}

# Function to make predictions
def predict_sentiment(text):
    start_time = timer()
    
    inputs = model.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    # Map predicted class to emoji
    predicted_class = torch.argmax(logits, dim=1).item()
    result = emoji_to_emotion[predicted_class]
    
    # Create a dictionary of class probabilities
    class_probabilities = {emoji_to_emotion[i]: float(probabilities[0, i]) for i in range(len(emoji_to_emotion))}
    
    # Calculate prediction time
    pred_time = round(timer() - start_time, 5)
    
    return class_probabilities, pred_time

# Create title, description and article strings
title = "Emoji-aware Sentiment Analysis using Roberta Model"
description = "Explore the power of sentiment analysis with our Emotion Detector! Simply input a sentence or text, and let our model predict the underlying emotion."
article = "Sentiment Analysis, also known as opinion mining, is a branch of Natural Language Processing (NLP) that involves determining the emotional tone behind a piece of text. This powerful tool allows us to uncover the underlying feelings, attitudes, and opinions expressed in written communication."


# Interface for Gradio
iface = gr.Interface(
    fn=predict_sentiment,
    inputs="text",
    outputs=[gr.Label(num_top_classes=7, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
    title=title,
    description=description,
    article=article)

# Launch the Gradio interface
iface.launch()
