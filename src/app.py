from flask import Flask, jsonify, request, render_template
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    question_csv = request.form.get('question')
    clientnum = request.form.get('clientnum')
    print(question_csv)

    predictions = pd.read_csv(f'predictions/{question_csv}')
    # Check if CLIENTNUM exists in the prediction data
    client_data = predictions[predictions['CLIENTNUM'] == int(clientnum)]
    
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404

    insights = {
        "CLIENTNUM": clientnum,
        "prediction": client_data.to_dict(orient='records')[0],
        "insights": None
    }
    
    return render_template('index.html', insights=insights)

if __name__ == '__main__':
    app.run(debug=True)