from flask import Flask, request, jsonify,  render_template, request, redirect, url_for
#from config.swagger import swagger_configuration

from model.prompt_retreiver import generate_response

app = Flask(__name__)
#swagger_configuration()


@app.post('/message')
def message():
    
    data = request.json
    
    print(data['query'])

    return generate_response(data['query'])
    #return jsonify(response_from_model)
    
@app.get('/index')
def hello():
    
    return 'Hello, World!'
