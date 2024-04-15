import base64
import io
from dotenv import load_dotenv
load_dotenv()

from e2b_code_interpreter import CodeInterpreter



def code_interpret( code):
    with CodeInterpreter() as sandbox:
        execution = sandbox.notebook.exec_cell(code)
        print(execution)

    return execution


code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('netflix.csv')

# Prepare the data
X = df.drop('type', axis=1)
y = df['type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)

# Find top 5 accuracy models
top_5_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)[:5]

# Choose the best model from the top 5
best_model_name, best_model_accuracy = top_5_models[0]
best_model = models[best_model_name]

# Model evaluation for the best model
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)
best_model_accuracy = accuracy_score(y_test, y_pred_best)

# Store the best model
import joblib
joblib.dump(best_model, 'best_model.pkl')

# Store the model building code
with open('ML_model.py', 'w') as file:
    file.write("# Put the code to build the best model here")


print("Top 5 models and their accuracies:")
for model_name, accuracy in top_5_models:
    print(f"{model_name}: {accuracy}")


print(f"\nBest model: {best_model_name}, Accuracy: {best_model_accuracy}")
"""

SYSTEM_PROMPT = """
## your job & context
you are a python data scientist. you are given tasks to complete and you run python code to solve them.
- the python code runs in jupyter notebook.
- every time you call `execute_python` tool, the python code is executed in a separate cell. it's okay to multiple calls to `execute_python`.
- display visualizations using matplotlib or any other visualization library directly in the notebook. don't worry about saving the visualizations to a file.
- you have access to the internet and can make api requests.
- you also have access to the filesystem and can read/write files.
- you can install any pip package (if it exists) if you need to but the usual packages for data analysis are already preinstalled.
- you can run any python code you want, everything is running in a secure sandbox environment.

## style guide
tool response values that have text inside "[]"  mean that a visual element got rended in the notebook. for example:
- "[chart]" means that a chart was generated in the notebook.
"""




# image = execution.results[0].png
#
#
# i = base64.b64decode(image)
# i = io.BytesIO(i)
# i = mpimg.imread(i, format='PNG')
#
# plt.imshow(i, interpolation='nearest')
# plt.show()