import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch
import pickle
from flask import Flask, request, jsonify, render_template

# Create flask app
flask_app = Flask(__name__)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_file = 'fine_tuned_bert_model.pkl'
label_encoder_file = 'label_encoder.pkl'
with open(model_file, 'rb') as f_model, open(label_encoder_file, 'rb') as f_encoder:
    model = pickle.load(f_model)
    label_encoder = pickle.load(f_encoder)

# Prediction function
def predict_category(text):
    tokens = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt', add_special_tokens=True)
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits

    predicted_class = torch.argmax(logits, dim=1).cpu().item()
    predicted_category = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_category

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    complaint_text = request.form['complaint']
    predicted_category = predict_category(complaint_text)
    return render_template("index.html", prediction_text = "The Complaint Category is {}".format(predicted_category))

if __name__ == "__main__":
    flask_app.run(debug=True)