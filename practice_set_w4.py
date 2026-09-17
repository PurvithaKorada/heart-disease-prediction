from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
import joblib
# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
print("Dataset loaded successfully")
# 2. Cross-validation
model = SVC()
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("\nCross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
# 3. GridSearchCV for SVM
parameters = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}
grid_search = GridSearchCV(
    SVC(),
    parameters,
    cv=5,
    scoring="accuracy"
)
grid_search.fit(X, y)
print("\nBest parameters:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)
# 4. Save the best trained model
best_model = grid_search.best_estimator_
joblib.dump(best_model, "svm_model.pkl")
print("\nModel saved successfully")
# 5. Load the saved model
loaded_model = joblib.load("svm_model.pkl")
print("Model loaded successfully")
# 6. Predict on new data
new_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = loaded_model.predict(new_data)
print("\nPrediction:", prediction)
print("Predicted flower:", iris.target_names[prediction[0]])

