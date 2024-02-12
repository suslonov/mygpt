import os
if  "__file__" in globals():
    os.sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    os.sys.path.append(os.path.abspath('.'))
    os.sys.path.append(os.path.dirname(os.path.abspath('.')))
from argparse import Namespace
from flask import Flask, render_template, request, jsonify, flash
from datetime import datetime, timedelta
import hashlib
import os
import time

STORAGE_FILE = "/www/ai-gpt/storage/0.txt"
application = Flask(__name__)
application.secret_key = 'random string'

@application.route('/45234hfkjsdf89j324h2kh432k4h2/<uuid>', methods = ['POST', 'GET'])
def home_page(uuid):
    if request.method == 'POST':
        text = request.get_json()
        with open(STORAGE_FILE, "a") as f:
            f.write(uuid)
            f.write(":")
            f.write(text)
            f.write("\n")
        return jsonify({'status': 'success'})
    else:
        with open(STORAGE_FILE, "r") as f:
            text = f.read()
        return jsonify({'status': 'success', "context": text})

