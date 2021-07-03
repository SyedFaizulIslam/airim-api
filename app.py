from flask import Flask,request,redirect,url_for,flash,jsonify
from flask_cors import CORS,cross_origin
import joblib
import numpy as np
import pandas as pd
import re

app= Flask(__name__)
cors=CORS(app)
app.config['CORS_HEADERS']='Content-Type'

def init():
    global model,vectorizer
    modelfile='ConsumerComplainModel.pkl'
    model=joblib.load(modelfile)

    vectorfile='ConsumerComplainTfidfVectorizer.pkl'
    vectorizer= joblib.load(vectorfile)

def clean_text(text):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """
    
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)    
    
    # convert text to lowercase
    text = text.strip().lower()
    
    # replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text
@app.route('/api/',methods=['GET'])
def helloworld():
    return "Hello World!!"

@app.route('/api/PredictBankProduct',methods=['POST'])
def PredictBankProduct():
    data=request.get_json()
    ConsumerComplain=data['ConsumerComplain']
    data = {'consumer_complaint_narrative':[ConsumerComplain]}
    result=ModelResult(data)
    return str(result)

def ModelResult(data):
    X=pd.DataFrame.from_dict(data)
    predict_features = vectorizer.transform(X["consumer_complaint_narrative"])
    y_pred = model.predict(predict_features)
    #result=np.round(prediction[0],2)
    return str(y_pred[0])

if __name__=='__main__':
    init()
    data = {'consumer_complaint_narrative':['My loans have been in payment regularly but now I need some time to pay my next EMI']}
    result = ModelResult(data)
    print('Model and vector initialize')
    print('Model Result:',result)
    app.run(host='0.0.0.0')
    #app.run(port='8080')h
