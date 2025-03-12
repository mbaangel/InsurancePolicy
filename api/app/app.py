from flask import Flask, request, jsonify,  render_template, request, redirect, url_for
#from config.swagger import swagger_configuration

#from model.embeddings import Embedding
#from model.prompt_retreiver import generate_response
from model.index import generate_response

app = Flask(__name__)
#swagger_configuration()


@app.post('/message')
def message():
    
    data = request.json
    
    #embedding = Embedding()
    #print(embedding.embedding_model)
    generate_response(data)
    return jsonify(data)
    
@app.get('/index')
def hello():
    
    return 'Hello, World!'
