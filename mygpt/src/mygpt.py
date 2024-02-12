import datetime
from flask import Flask, render_template, request, flash
from hashlib import md5
import json
from openai import OpenAI


STORAGE_FILES = "/www/mygpt/storage/"
OPENAI_KEY_FILE = "/www/mygpt/keys/chat_gpt_anton.sec"
SECRET_URL = "/www/mygpt/keys/secret_url.sec"

with open(OPENAI_KEY_FILE, 'r') as f:
    openai_key = f.read().strip()

with open(SECRET_URL, 'r') as f:
    secret_url = f.read().strip()

application = Flask(__name__)
application.secret_key = 'random string'
sub_path = "/"

GREETINGS = "hi, I'm chat GPT bot"

@application.route(sub_path + ("/" + secret_url + "/" if secret_url else "") + "id/<uuid>", methods = ['GET'])
def home_page_wd1(uuid):
    return _home_page(uuid)

@application.route(sub_path + ("/" + secret_url if secret_url else ""), methods = ['POST', 'GET'])
def home_page_wd2():
    return _home_page(None)

def _home_page(uuid):
    if request.method == 'POST':
        result = request.form
        re = result.get("wd_chatinput")
        sessionid = result.get("session_id")
        file_name = STORAGE_FILES + md5(sessionid.encode("utf8")).hexdigest()
        try:
            with open(file_name, "r") as f:
                messages = json.load(f)
            if len(messages) > 0 and "session_id" in messages[-1]:
                messages.pop()
        except:
            messages = []
        
        if re:
            messages.append({"role": "user", "content": re})
            client = OpenAI(api_key=openai_key)
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                temperature=0,
                # response_format={ "type": "json_object" },
                messages=messages
            )
            messages.append({"role": "system", "content": completion.choices[0].message.content})
            messages.append({"session_id": sessionid})
            messages.pop()
            with open(file_name, "w") as f:
                json.dump(messages, f)
        return render_template('home-page-mygpt.html',
                               sub_path = sub_path,
                               greetings = GREETINGS,
                               session_id = sessionid,
                               messages = messages)
    else:
        if uuid:
            sessionid = uuid
            file_name = STORAGE_FILES + md5(sessionid.encode("utf8")).hexdigest()
            with open(file_name, "r") as f:
                messages = json.load(f)
            if len(messages) > 0 and "session_id" in messages[-1]:
                messages.pop()
        else:
            sessionstr = request.environ.get('HTTP_X_REAL_IP', request.remote_addr) + str(datetime.datetime.now())
            sessionid = md5(sessionstr.encode("utf8")).hexdigest()
            messages = []
        return render_template('home-page-mygpt.html',
                               sub_path = sub_path,
                               greetings = GREETINGS,
                               session_id = sessionid,
                               messages = messages)

