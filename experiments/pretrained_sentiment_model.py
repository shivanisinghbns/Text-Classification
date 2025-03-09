from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class huggingface_model:
    def __init__(self):
        self.model_name = "tabularisai/multilingual-sentiment-analysis"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def predict_sentiment_pretrained(self,text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
        return [sentiment_map[p] for p in torch.argmax(probabilities, dim=-1).tolist()]
    
# pretrained_model=huggingface_model()
# print(pretrained_model.predict_sentiment_pretrained(text="I love the weather today"))
    

