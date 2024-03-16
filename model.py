import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import torch,pickle


# Convert data to DataFrame
df = pd.read_csv('ticketClassifier.csv')

# Step 1: Label Encoding
label_encoder = LabelEncoder()
df['complaint_title_encoded'] = label_encoder.fit_transform(df['complaint_title'])

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['complaint_title_encoded'], test_size=0.2, random_state=42)

# Step 3: Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize text data
X_train_tokens = tokenizer(list(X_train), padding=True, truncation=True, max_length=128, return_tensors='pt', add_special_tokens=True)
X_test_tokens = tokenizer(list(X_test), padding=True, truncation=True, max_length=128, return_tensors='pt', add_special_tokens=True)

# Convert target labels to LongTensor explicitly
y_train_tensor = torch.LongTensor(y_train.values)
y_test_tensor = torch.LongTensor(y_test.values)

# Step 4: Create DataLoader
train_dataset = TensorDataset(X_train_tokens['input_ids'], X_train_tokens['attention_mask'], y_train_tensor)
test_dataset = TensorDataset(X_test_tokens['input_ids'], X_test_tokens['attention_mask'], y_test_tensor)

batch_size = 2  # Adjust batch size as per your system's memory

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Step 5: Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['complaint_title'].unique()))

# Step 6: Fine-tune BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

num_epochs = 25  # You can adjust the number of epochs
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss}")

# Step 7: Evaluate the model
model.eval()
preds = []
true_labels = []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, preds)
print(f"Accuracy: {accuracy}")
precision = precision_score(true_labels, preds,average='macro')
print(f"precision: {precision}")
recall = recall_score(true_labels, preds,average='macro')
print(f"recall: {recall}")

# Step 5: Save the fine-tuned model and LabelEncoder as a pickle file
output_model_file = "fine_tuned_bert_model.pkl"
label_encoder_file = "label_encoder.pkl"
with open(output_model_file, 'wb') as f_model, open(label_encoder_file, 'wb') as f_encoder:
    pickle.dump(model, f_model)
    pickle.dump(label_encoder, f_encoder)

print("Model and LabelEncoder saved as pickle files.")

print("Model saved as pickle file.")

# Step 6: Load the model from the pickle file for inference
with open(output_model_file, 'rb') as f:
    loaded_model = pickle.load(f)

print("Model loaded from pickle file.")

# Step 8: Prediction on new complaint
def predict_category(text):
    tokens = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt', add_special_tokens=True)
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    with torch.no_grad():
        output = loaded_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits

    predicted_class = torch.argmax(logits, dim=1).cpu().item()
    predicted_category = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_category

# Test prediction on new complaint
new_complaint = "recently open citibank citigold checking account advertise signup bonus aadvantage point upon completion two consecutive bill payment debit card purchase sign intent complete requirement representative confirm offer apply new confirmation communicate via online secure message feature attach copy reference direct inquire expect delivery date earn tell target would receive promise conflict previously already spend considerable amount time meet"
predicted_category = predict_category(new_complaint)
print(f"Predicted category for '{new_complaint}': {predicted_category}")