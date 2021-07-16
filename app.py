import numpy as np
from flask import Flask, request, render_template
from joblib import load
import joblib
from tensorflow.keras.models import load_model 
from sklearn.feature_extraction.text import CountVectorizer
import keras
import tensorflow as tf

tf.keras.backend.clear_session()
app = Flask(__name__)
model=tf.keras.models.load_model(r"D:/internship/Flask/amazo.h5")
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    d = request.form['Sentence']
    print(d)
    loaded=CountVectorizer(decode_error='replace',vocabulary=joblib.load(r"D:\internship\Flask\amazo.save"))
    d=d.split("delimiter")
    result=model.predict(loaded.transform(d))
    print(result)
    prediction=result>0.5
    if prediction[0] == False:
    	output="Positive review"
    elif prediction[0] == True:
    	output="Negative review"
    return render_template('index.html', prediction_text='{}'.format(output))
if __name__ == "__main__":
    app.run(debug=True)
