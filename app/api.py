from flask import Flask, render_template, request
from flask_restful import Api, Resource
try:
    from .model import BERTModel
except ImportError:  # pragma: no cover - fallback for script execution
    from model import BERTModel

app = Flask(__name__,template_folder='../templates')
api = Api(app)
bert_model = BERTModel()

@app.route('/')
def home():
    return render_template('index.html')

class Predict(Resource):
    def post(self):
        data = request.get_json(force=True) or {}
        texts = data.get('texts')
        if not isinstance(texts, list):
            return {'message': "'texts' field is required and must be a list"}, 400

        embeddings = bert_model.get_embeddings(texts)
        return {'embeddings': embeddings}

class WordEmbeddings(Resource):
    def post(self):
        data = request.get_json(force=True) or {}
        text = data.get('text')
        if not isinstance(text, str):
            return {'message': "'text' field is required and must be a string"}, 400

        tokens, embeddings = bert_model.get_word_embeddings(text)
        return {'tokens': tokens, 'embeddings': embeddings}

class Sentiment(Resource):
    def post(self):
        data = request.get_json(force=True) or {}
        texts = data.get('texts')
        if not isinstance(texts, list):
            return {'message': "'texts' field is required and must be a list"}, 400

        sentiments = bert_model.analyze_sentiment(texts)
        return {'sentiments': sentiments}

class Entities(Resource):
    def post(self):
        data = request.get_json(force=True) or {}
        texts = data.get('texts')
        if not isinstance(texts, list):
            return {'message': "'texts' field is required and must be a list"}, 400

        entities = bert_model.extract_entities(texts)
        return {'entities': entities}


class Summarize(Resource):
    def post(self):
        data = request.get_json(force=True) or {}
        texts = data.get('texts')
        if not isinstance(texts, list):
            return {'message': "'texts' field is required and must be a list"}, 400

        summaries = bert_model.summarize_texts(texts)
        return {'summaries': summaries}

@app.route('/health')
def health():
    return {
        'status': 'ok',
        'model_loaded': bert_model.is_model_loaded(),
        'model_name': bert_model.model_name,
    }


api.add_resource(Predict, '/api/predict')
api.add_resource(WordEmbeddings, '/api/word_embeddings')
api.add_resource(Sentiment, '/api/sentiment')
api.add_resource(Entities, '/api/entities')
api.add_resource(Summarize, '/api/summarize')

if __name__ == '__main__':
    app.run(debug = True,host='0.0.0.0', port=8888)
