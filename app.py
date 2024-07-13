from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # Make a prediction using the loaded model
        prediction = model.predict([[feature1, feature2]])

        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
