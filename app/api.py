from flask import Flask, render_template
from flask_restful import Api, Resource, reqparse
from model import BERTModel

app = Flask(__name__,template_folder='../templates')
api = Api(app)
bert_model = BERTModel()

@app.route('/')
def home():
    return render_template('index.html')

class Predict(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('texts', type=list, location='json')
        args = parser.parse_args()
        input_texts = args['texts']

        embeddings = bert_model.get_embeddings(input_texts)

        return {'embeddings': embeddings}

class WordEmbeddings(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('text', type=str, location='json')
        args = parser.parse_args()
        input_text = args['text']

        tokens, embeddings = bert_model.get_word_embeddings(input_text)

        return {'tokens': tokens, 'embeddings': embeddings}

class Sentiment(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('texts', type=list, location='json')
        args = parser.parse_args()
        input_texts = args['texts']

        sentiments = bert_model.analyze_sentiment(input_texts)

        return {'sentiments': sentiments}

class Entities(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('texts', type=list, location='json')
        args = parser.parse_args()
        input_texts = args['texts']

        entities = bert_model.extract_entities(input_texts)

        return {'entities': entities}

api.add_resource(Predict, '/api/predict')
api.add_resource(WordEmbeddings, '/api/word_embeddings')
api.add_resource(Sentiment, '/api/sentiment')
api.add_resource(Entities, '/api/entities')

if __name__ == '__main__':
    app.run(debug = True,host='0.0.0.0', port=8888)
