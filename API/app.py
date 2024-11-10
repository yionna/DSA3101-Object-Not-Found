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

    predictions = pd.read_csv(f'../data/predictions/{question_csv}')
    client_data = predictions[predictions['CLIENTNUM'] == int(clientnum)]
    
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404

    insights = {
        "CLIENTNUM": clientnum,
        "headers": list(predictions.columns),
        "prediction": client_data.to_dict(orient='records')[0],
        "insights": None
    }
    
    return render_template('index.html', insights=insights)

if __name__ == '__main__':
    app.run(debug=True)