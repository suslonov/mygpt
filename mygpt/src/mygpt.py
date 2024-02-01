from os import listdir
import string
import datetime
import random
from flask import Flask, render_template, request, flash, session
from hashlib import md5
import MySQLdb
import json

application = Flask(__name__)
application.secret_key = 'random string'
sub_path = "/"
predict_len = 4
predictions = ['standard','suspicious', 'fraud']

csv_examples_path = "csv_examples"
sms_examples_path = "sms_examples"

with open(csv_examples_path, 'r') as f:
  csv_examples_list = f.read().splitlines()
with open(sms_examples_path, 'r') as f:
  sms_examples_list = f.read().splitlines()

 
db = MySQLdb.connect(host="localhost", user="watchdog", passwd="watchdog_proto", db="watchdog_proto")
mycur = db.cursor()

#catlist_path = "catlist.csv"
#save_path = "saves"
#log_path = "logs/request_log"
#catlist = util.read_csv_to_tuples(catlist_path)
#saves = [f for f in listdir(save_path)]
#saves.sort(key=lambda k:(k[1],k[3],k[4]))
#saves_l = list(map(util.unpack_name, saves))
#saves_l.sort(key=lambda k:(k[1],k[3],k[4]))
#nn_list = list(enumerate(saves_l))
#translator = str.maketrans('', '', string.punctuation)

def Proto_Save_DB(s1, s2, t1):
  s = """INSERT INTO watchdog_data (sessionid, rowname, actual, wdata) VALUES (%s, %s, %s, %s)"""
  try:
    mycur.execute(s, (s1, s2, 1, t1))
  except Exception:
    db = MySQLdb.connect(host="localhost", user="watchdog", passwd="watchdog_proto", db="watchdog_proto")
    mycur = db.cursor()
    mycur.execute(s, (s1, s2, 1, t1))
    
  db.commit()

def Proto_Load_DB(s1, s2):
  s = """select * from watchdog_data where (sessionid = %s) and (rowname = %s) order by id desc"""

  try:
    i = mycur.execute(s, (s1, s2))
  except Exception:
    db = MySQLdb.connect(host="localhost", user="watchdog", passwd="watchdog_proto", db="watchdog_proto")
    mycur = db.cursor()
    i = mycur.execute(s, (s1, s2))

  if i > 0:
    return mycur.fetchall()[0]
  else:
    return ()


def Proto_Predict(inputtext, sessionid):
  predict_list=[]
  for i in range(predict_len):
    predict_list1=[]
    for j in range(len(predictions)):
      predict_list1.append(random.random())
    predict_list1[0] += 0.4
    norm = sum(predict_list1)
    predict_max = 0
    j_max = 0
    for j in range(len(predictions)):
      predict_list1[j] = predict_list1[j]/norm
      if predict_max < predict_list1[j]:
        predict_max = predict_list1[j]
        j_max = j
#    store predict_list1
    predict_i = (j_max, predictions[j_max], predict_max, i)
    predict_list.append(predict_i)
  predict_max = 0

  p = max(predict_list, key = lambda x: x[0])

  si = 0
  sp = 0
  
  for i in range(predict_len):
    if p[0] == predict_list[i][0]:
      si = si + 1
      sp = sp + predict_list[i][2]
  p_avg = sp / si

  fake_s = round(random.random()*1000, 2)
  if p[0] == 0:
    s = "I think the payment {:,.2f} is standard payment. You've paid for goods in a local shop.".format(fake_s)
  elif p[0] == 1:
    s = "I think the payment {:,.2f} is suspicious. You've paid for goods in a unknown online shop.".format(fake_s)
  else:
    s = "I think the payment {:,.2f} is possible card fraud. Somebody used your card to pay for car tires in Romania!".format(fake_s)

  predict = (s, p[0], p_avg, predict_list, [fake_s, 100, 200])
  Proto_Save_DB(sessionid, 'predict', json.dumps(predict))
  return predict 

@application.route(sub_path, methods = ['POST', 'GET'])
def home_page_wd():
  if request.method == 'POST':
    result = request.form
    re = result.get("wd_testinput")
    if re:
      sessionid = session['uid']
      Proto_Save_DB(sessionid, 'input', re)
      predict = Proto_Predict(re, sessionid)
      return render_template('home-page-wd.html', sub_path=sub_path, csv_examples=csv_examples_list, sms_examples=sms_examples_list, predict=predict, predict1=predict[1])
    else:
      return render_template('home-page-wd.html', sub_path=sub_path, csv_examples=csv_examples_list, sms_examples=sms_examples_list)
  else:
    sessionstr = request.environ.get('HTTP_X_REAL_IP', request.remote_addr) + str(datetime.datetime.now())
    sessionid = md5(sessionstr.encode("utf8")).hexdigest()
    session['uid'] = sessionid
    return render_template('home-page-wd.html', sub_path=sub_path, csv_examples=csv_examples_list, sms_examples=sms_examples_list)

@application.route('/details/<int:detail_n>', methods = ['POST', 'GET'])
def home_page_wd_d(detail_n):
  i = detail_n
  if request.method == 'POST':
    pass
#    result = request.form
#    re = result.get("wd_testinput")
#    sessionid = session['uid']
#    Proto_Save_DB(sessionid, 'input', re)
#    predict = Proto_Predict(re, sessionid)
#
#    return render_template('home-page-wd.html', sub_path=sub_path, csv_examples=csv_examples_list, sms_examples=sms_examples_list, predict=predict)
  else:
    sessionid = session['uid']
    ps = Proto_Load_DB(sessionid, 'predict')
    predict = json.loads(ps[4])
    return render_template('home-page-wd.html', sub_path=sub_path, csv_examples=csv_examples_list, sms_examples=sms_examples_list, sessionid=sessionid, predict_t=ps[4], predict_i=str(predict[3][i]), predict1=int(predict[1]))

@application.route('/chat/<int:p1>', methods = ['POST', 'GET'])
def home_page_wd_chat(p1):
  sessionid = session['uid']
  ps = Proto_Load_DB(sessionid, 'chat')
  if ps:
    psl = json.loads(ps[4])
  else:
    psl = []
  if request.method == 'POST':
    result = request.form
    re = result.get("wd_chatinput")
    if re:
      psl.append((0, re))
  if psl:
    if psl[-1][0] == 0:
      if len(psl) > 3:
        psl.append((1, "Sorry, I can't speak. I'm just tiny fox"))
      elif len(psl) > 1:
        psl.append((1, "I'm sorry, my creator promised to learn me to speak but still didn't do it. I'm just tiny desert fox"))
  else:
    dt = datetime.datetime.now()
    if dt.hour >= 4 and dt.hour < 11:
      dtt = "morning"
    elif dt.hour >= 11 and dt.hour < 18:
      dtt = "afternoon"
    elif dt.hour >= 18 and dt.hour < 23:
      dtt = "evening"
    else:
      dtt = "night"
    psl.append((1, "Good "+dtt+", I am Fennec"))

  Proto_Save_DB(sessionid, 'chat', json.dumps(psl))
  return render_template('home-page-wd.html', sub_path=sub_path, csv_examples=csv_examples_list, sms_examples=sms_examples_list, chat=psl, predict1=p1)

#@application.route("/nn/log")
#def log_page_nn():
#  with open(log_path, 'r') as f:
#    ff=f.readlines()
#  
#  fff = []  
#  for fl in ff:
#    try:
#      fff.append(fl.split('|')[2])
#    except Exception:
#      pass
#
#  return render_template('log-nn.html', fff=fff)
#  



#    if not sentence:
#      return render_template('home-page-nn.html', sub_path=sub_path, sentence=sentence, nn_list=nn_list, catlist=catlist)
#    with open(log_path, 'a') as f:
#      f.write(str(datetime.datetime.now()) + "|" + request.environ.get('HTTP_X_REAL_IP', request.remote_addr) + "|" + sentence + '\n')
#    sentence = sentence.lower()
#    sentence = sentence.translate(translator)
#    
#    nn_checked_list=[]
#    for res in result:
#      if res[0:3]=="nn_":
#        nn_checked_list.append(int(res[3:]))
#      if len(nn_checked_list)>2: break
#    
#    nn_output_names=[]
#    nn_output_lists=[]
#    
#    for nncl in nn_checked_list:
#      nn_output_names.append(saves[nncl])
#      
#      config = util.TrainConfig(dataset=nn_list[nncl][1][0],
#                         nn_type=nn_list[nncl][1][1],
#                         input_length=nn_list[nncl][1][2],
#                         num_layers=nn_list[nncl][1][3],
#                         hidden_size=nn_list[nncl][1][4],
#                         num_labels=nn_list[nncl][1][5],
#                         keep_prob=nn_list[nncl][1][6],
#                         learning_rate=nn_list[nncl][1][7],
#                         lr_decay_start=nn_list[nncl][1][8],
#                         lr_decay_rate=nn_list[nncl][1][9],
#                         number_epochs=nn_list[nncl][1][10],
#                         save_path=save_path)
#
#      out_list = cat_enquirer.cat_model_enquarer([sentence], ['1'], catlist, config)     # suppose label 1 exists !!!
#      nn_output_lists.append(out_list[0][2])
