from json import dumps as encode_json
from flask import Flask, request
from parser import add_link_to_csv

app = Flask(__name__)


@app.post('/api/auth')
def auth():
    data = request.json
    login = data['login']
    password = data['password']
    
    if login == 'admin' and password == 'admin':    
        return encode_json({
            'status': True,
            'userID': '7aa2943e-473e-11ed-b878-0242ac120002'
        })

    return encode_json({
        'status': False
    })

    
@app.post('/api/sendData')
def sendData():
    data = request.json
    link = data['link']
    roles = data['roles']
    
    # print(link)
    # print(roles)
    
    add_link_to_csv(link, roles)
    
    return {'status': True}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)