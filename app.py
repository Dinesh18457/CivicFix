import os
import joblib
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, session

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(BASE_DIR, 'civicfix_app', 'templates')
static_dir = os.path.join(BASE_DIR, 'civicfix_app', 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = 'civicfix_secret_key'  # Required for saving history

MODELS_DIR = os.path.join(BASE_DIR, 'civicfix_app', 'models')
if not os.path.exists(MODELS_DIR):
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
agency_model = joblib.load(os.path.join(MODELS_DIR, 'agency_routing_model.pkl'))
resolution_model = joblib.load(os.path.join(MODELS_DIR, 'resolution_time_model.pkl'))

@app.route('/', methods=['GET'])
def index():
    # Retrieve all previous submissions
    history = session.get('history', [])
    return render_template('index.html', history=history)

@app.route('/predict', methods=['POST'])
def predict():
    descriptor = request.form.get('descriptor', '')
    borough = request.form.get('borough', '')
    current_hour = datetime.now().hour

    input_df = pd.DataFrame([{
        'descriptor': descriptor,
        'borough': borough,
        'created_hour': current_hour
    }])

    transformed_input = preprocessor.transform(input_df)
    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()

    pred_agency = agency_model.predict(transformed_input)[0]
    pred_time = float(resolution_model.predict(transformed_input)[0])
    pred_time_formatted = f"{max(0.5, pred_time):.1f}"

    # Append new submission to the history list
    history = session.get('history', [])
    history.insert(0, {
        'id': len(history) + 1,
        'descriptor': descriptor,
        'borough': borough,
        'agency': pred_agency,
        'time': pred_time_formatted
    })
    session['history'] = history  # Save back to session

    return render_template('index.html', history=history, borough=borough)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
