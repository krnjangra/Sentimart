from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return jsonify({'sentiment': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)