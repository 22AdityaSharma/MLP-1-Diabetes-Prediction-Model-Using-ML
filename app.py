from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), 'diabetes_model.pkl')

with open(model_path, 'rb') as file:
    model = pickle.load(file)


# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all form values
        features = [float(x) for x in request.form.values()]

        # Convert into numpy array
        final_features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_features)

        # Result
        if prediction[0] == 1:
            output = "Diabetic"
        else:
            output = "Not Diabetic"

        return render_template(
            'index.html',
            prediction_text=f'Prediction Result: {output}'
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}'
        )


if __name__ == "__main__":
    app.run(debug=True)