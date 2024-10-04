import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model = load_model('cnn_model.h5')

# Load the data and create the scaler
data = pd.read_csv('vldlp_hscrp_data.csv')
scaler = StandardScaler()
X = data[['VLDLP_C', 'hsCRP']].values
scaler.fit(X)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("VLDLP-C and hsCRP Predictor"),
    html.Label("VLDLP-C level"),
    dcc.Input(id='vldlp_input', type='number', value=50),
    html.Label("hsCRP level"),
    dcc.Input(id='hscrp_input', type='number', value=2),
    html.Button('Predict', id='predict-button', n_clicks=0),
    html.Div(id='prediction-output')
])

# Define the callback
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('vldlp_input', 'value'), Input('hscrp_input', 'value')]
)
def predict_vldlp_hscrp(n_clicks, vldlp_value, hscrp_value):
    if n_clicks > 0:
        input_data = np.array([[vldlp_value, hscrp_value]])
        input_data = scaler.transform(input_data)
        input_data = input_data.reshape((input_data.shape[0], 2, 1))

        prediction = model.predict(input_data)
        result = "High Risk" if prediction > 0.5 else "Low Risk"
        return f"Prediction: {result}"
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)

