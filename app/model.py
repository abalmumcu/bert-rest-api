import torch
from transformers import BertTokenizer, BertModel, pipeline

class BERTModel:
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

        # Use task-specific pretrained models for sentiment analysis and NER instead
        # of the base ``BertModel``.  Passing the raw ``BertModel`` to these
        # pipelines raises errors because it lacks the classification heads
        # required for the tasks.
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.entity_extractor = pipeline("ner")

        # Lazily load the base model only when embedding utilities are used.

    def _ensure_base_model(self):
        if self.model is None or self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            self.model.eval()

    def get_embeddings(self, input_texts):
        self._ensure_base_model()
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
        # ``outputs.last_hidden_state`` already has shape
        # (batch_size, sequence_length, hidden_size). Converting directly to a
        # list preserves the batch dimension for single and multi-sample
        # inputs alike.
        embeddings = outputs.last_hidden_state.tolist()

        return embeddings

    def tokenize_text(self, input_text):
        self._ensure_base_model()
        tokens = self.tokenizer.tokenize(input_text)
        return tokens

    def get_word_embeddings(self, input_text):
        self._ensure_base_model()
        tokens = self.tokenizer.tokenize(input_text)
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
