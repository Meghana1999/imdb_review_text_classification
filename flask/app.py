from flask import Flask, request, render_template
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)

# Loading the trained model 
model_path = "/Users/meghana/Desktop/imdb_final/saved_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)
    return 'positive' if prediction.item() == 1 else 'negative'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        sentiment = predict_sentiment(text)
        return render_template('index.html', sentiment=sentiment)
    return render_template('index.html', sentiment='')

if __name__ == '__main__':
    app.run(debug=True)



















# from google.colab import drive
# drive.mount('/content/drive')

# !pip install datasets
# !pip install transformers

# df = pd.read_csv("/content/drive/MyDrive/imdb/IMDB Dataset.csv")

# df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})


# # Function for basic preprocessing
# def preprocess_text(text):
#     # Convert to lowercase
#     text = text.lower()
#     # Remove HTML tags
#     text = re.sub(r'<.*?>', '', text)
#     # Remove special characters and punctuation
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word not in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     return ' '.join(tokens)




# df['cleaned_text'] = df['review'].apply(preprocess_text)

# train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# # Create Hugging Face Datasets for train and test sets
# train_dataset = Dataset.from_pandas(train_df)
# test_dataset = Dataset.from_pandas(test_df)

# # Tokenization and model setup
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)



# def tokenize_text_data(sample_text):
#     return tokenizer(sample_text['cleaned_text'], padding="max_length", truncation=True, max_length=512)

# # Apply tokenization
# tokenized_train_dataset = train_dataset.map(tokenize_text_data, batched=True)
# tokenized_test_dataset = test_dataset.map(tokenize_text_data, batched=True)

# # Access the train and test sets
# train_set = tokenized_train_dataset
# test_set = tokenized_test_dataset




# # Metrics for evaluation
# def eval_metrics(eval_pred):
#     metric = load_metric("f1")
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

# # Training arguments
# training_args = TrainingArguments(
#     output_dir="/content/drive/MyDrive/imdb/results2",
#     num_train_epochs=3,
#     per_device_train_batch_size=24,
#     per_device_eval_batch_size=24,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     fp16=True,
#     metric_for_best_model="f1"
# )

# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_set,
#     eval_dataset=test_set,
#     compute_metrics=eval_metrics
# )

# # Train the model
# trainer.train()


# model_path = "/content/drive/MyDrive/imdb/saved_model2"
# model.save_pretrained(model_path)
# tokenizer.save_pretrained(model_path)





# # Function to format the batches correctly
# def collate_fn(batch):
#     # Convert list of token IDs to tensors before stacking
#     return {
#         'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
#         'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
#         'labels': torch.tensor([item['label'] for item in batch])
#     }

# # Function to measure prediction speed
# def benchmark_prediction_speed(model, dataset, batch_size=16):
#     # DataLoader for the dataset with the custom collate function
#     data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

#     # Place model in evaluation mode
#     model.eval()

#     # Tracking total prediction time
#     start_time = time.time()

#     # Iterate over data_loader and make predictions
#     for batch in data_loader:
#         # Move batch to the same device as model
#         batch = {k: v.to(model.device) for k, v in batch.items()}

#         with torch.no_grad():
#             outputs = model(**batch)

#     # Calculate total time taken
#     total_time = time.time() - start_time
#     return total_time

# # Combine train and test datasets for benchmarking
# full_dataset = torch.utils.data.ConcatDataset([train_set, test_set])

# # Benchmark prediction speed
# total_prediction_time = benchmark_prediction_speed(model, full_dataset)
# print(f"Total time taken for predictions: {total_prediction_time:.2f} seconds")



























