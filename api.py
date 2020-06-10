import flask
import gensim

app = flask.Flask(__name__)

app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
  return "<h1>Hello world !</h1>"

@app.route('/api/inferVector', methods=['GET'])
def infer_vector():
  if not flask.request.json or not 'text' in flask.request.json:
    flask.abort(400)
  tokens = gensim.utils.simple_preprocess(flask.request.json['text'])
  model = gensim.models.Doc2Vec.load('tagging-model.d2v')
  vector = model.infer_vector(tokens)
  return flask.jsonify({'vector':vector.tolist()}), 200

app.run()