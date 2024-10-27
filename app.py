from flask import Flask, request, jsonify,render_template
import pandas as pd
import pickle

app = Flask(__name__)
app.static_folder = 'static'
@app.route('/')
def home():
    return render_template('welcome.html')


@app.route('/startprediction',methods=['POST'])
def startprediction():
    return render_template('home.html')



@app.route('/predict', methods=['POST'])
def predict():
    #model = pickle.load(open("model.pkl","rb"))
    b = request.form.to_dict()
    print(b)
    
    model = pickle.load(open("model.pkl","rb")) 
    p = model.predict(b)
    return render_template('p.html',p_text = f"Prediction = {p[0]}  ")



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
