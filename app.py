from flask import Flask, request, jsonify, render_template
from experiments.pretrained_sentiment_model import huggingface_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/pretrained', methods=['POST'])
def predict_pretrained():
    pretrained_model = huggingface_model()

    # Try getting text from form
    text = request.form.get('text')

    # If form data is empty, check for JSON request
    if not text:
        data = request.get_json()
        if data:
            text = data.get('text')

    if not text:
        return jsonify({"error": "Text is required!"}), 400

    result = pretrained_model.predict_sentiment_pretrained(text)
    print(result)
    return jsonify({"sentiment": result[0]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
