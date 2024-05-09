# from flask import Flask, request, jsonify
# import pandas as pd
# import xlrd
# import joblib

# app = Flask(__name__)

# # Load trained models
# models = {
#     "random_forest": joblib.load("RF_model.pkl"),
#     "decision_tree": joblib.load("dt_classifier.pkl"),
#     "stacking_ensemble": joblib.load("stacking_model.pkl"),
#     "voting_ensemble": joblib.load("voting_model.pkl"),
#     "bagging_ensemble": joblib.load("bagging_model.pkl"),
#     "svm": joblib.load("svm_model.pkl")
# }

# def predict(model_name, data):
#     model = models[model_name]
#     predictions = model.predict(data)
#     # Perform any further processing here if needed
#     return predictions

# @app.route('/predict', methods=['POST'])
# def handle_prediction():
#     # Get model name and XLS file from request
#     model_name = request.form['model']
#     xls_file = request.files['xls_file']
    
#     # Load XLS file
#     data = pd.read_excel(xls_file)
    
#     # Perform prediction
#     predictions = predict(model_name, data)
    
#     # Example output, modify as needed
#     output = {
#         "model_name": model_name,
#         "total_instances": len(data),
#         "correct_instances": predictions.sum(),
#         "incorrect_instances": len(data) - predictions.sum(),
#         "training_percentage": None  # Add training percentage calculation
#     }
    
#     return jsonify(output)

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load trained models
models = {
    "random_forest": joblib.load("RF_model.pkl"),
    "decision_tree": joblib.load("dt_classifier.pkl"),
    "stacking_ensemble": joblib.load("stacking_model.pkl"),
    "voting_ensemble": joblib.load("voting_model.pkl"),
    "bagging_ensemble": joblib.load("bagging_model.pkl"),
    "svm": joblib.load("svm_model.pkl")
}

def predict(model_name, data):
    model = models[model_name]
    predictions = model.predict(data)
    # Perform any further processing here if needed
    return predictions

def preprocess_data(data):
    # Strip leading and trailing whitespace, capitalize strings
    #true_label=[]
    data = data.drop(data.columns[0], axis=1)
    data['fetal_Health'] = data['fetal_Health'].str.strip().str.capitalize()
    # Define category mapping
    category_mapping = {
        'Suspicion': 2,
        'Normal': 1,
        'Pathologic': 3
    }
    # Apply mapping to the DataFrame
    data['fetal_Health'] = data['fetal_Health'].map(category_mapping)
    return data

# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    # Get model name and XLS file from request
    model_name = request.form['model']
    xls_file = request.files['xls_file']
    
    # Load XLS file
    data = pd.read_excel(xls_file)
    # print(data.columns)
    data = preprocess_data(data)
    true_label=data["fetal_Health"]
    data=data.drop(["fetal_Health"],axis=1)
    # print("/n")
    # print(data.columns)
    
    # Perform prediction
    predictions = predict(model_name, data)

    correct_instances = sum(1 for true_label, prediction in zip(true_label, predictions) if true_label == prediction)
    
    # Example output, modify as needed
    output = {
        "model_name": model_name,
        "total_instances": len(data),
        "correct_instances": correct_instances,
        "incorrect_instances": len(data) - correct_instances,
        "training_percentage": None  # Add training percentage calculation
    }
    output = {key: int(value) if isinstance(value, (int, np.int64)) else value for key, value in output.items()}
    print(model_name)
    print(len(data))
    print(predictions.sum())
    print(len(data-predictions.sum()))
    
    
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)

