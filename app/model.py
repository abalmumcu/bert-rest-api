import torch
from transformers import BertTokenizer, BertModel, pipeline

class BERTModel:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

        self.sentiment_analyzer = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        self.entity_extractor = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def get_embeddings(self, input_texts):
        encoded_inputs = self.tokenizer.batch_encode_plus(
            input_texts,
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state.squeeze(0)
        embeddings = embeddings.tolist()

        return embeddings

    def tokenize_text(self, input_text):
        tokens = self.tokenizer.tokenize(input_text)
        return tokens

    def get_word_embeddings(self, input_text):
        tokens = self.tokenize_text(input_text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([input_ids])

        with torch.no_grad():
            outputs = self.model(input_ids)

        embeddings = outputs.last_hidden_state.squeeze(0)
        embeddings = embeddings.tolist()

        return tokens, embeddings

    def analyze_sentiment(self, input_texts):
        return self.sentiment_analyzer(input_texts)

    def extract_entities(self, input_texts):
        return self.entity_extractor(input_texts)
