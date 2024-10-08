from flask import Flask, request, jsonify,render_template
import pickle

app = Flask(__name__)

@app.route('/')

def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    model = pickle.load(open("model.pkl","rb"))
    a = [int(x) for x in request.form.values()]
    pp = model.predict_proba([a])
    p = model.predict([a])
    
    return render_template('p.html',p_text=f"prediction = {p} and probability = {pp[0,0]}")



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

