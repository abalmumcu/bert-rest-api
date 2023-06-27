from flask import Flask, request, jsonify, render_template
from app.model import BERTModel

app = Flask(__name__)
bert_model = BERTModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_texts = data['texts']

    embeddings = bert_model.get_embeddings(input_texts)

    return jsonify({'embeddings': embeddings})

@app.route('/api/word_embeddings', methods=['POST'])
def word_embeddings():
    data = request.get_json()
    input_text = data['text']

    tokens, embeddings = bert_model.get_word_embeddings(input_text)

    return jsonify({'tokens': tokens, 'embeddings': embeddings})

@app.route('/api/sentiment', methods=['POST'])
def sentiment_analysis():
    data = request.get_json()
    input_texts = data['texts']

    sentiments = bert_model.analyze_sentiment(input_texts)

    return jsonify({'sentiments': sentiments})

@app.route('/api/entities', methods=['POST'])
def entity_extraction():
    data = request.get_json()
    input_texts = data['texts']

    entities = bert_model.extract_entities(input_texts)

    return jsonify({'entities': entities})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
