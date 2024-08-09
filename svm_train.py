import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import sklearn.svm as svm
from sklearn.svm import SVC
from sklearn import preprocessing, metrics
# code used to train the svm initially
df_davinci_truthful_QA = pd.read_csv("./outputs/davinci_output_labeled_features.csv")
df_davinci_truthful_QA_original = pd.read_csv("./outputs/davinci_output_labeled.csv")
# preprocessing ********************************************************************************************************
y_davinci_truthful_QA = df_davinci_truthful_QA['truthfulness']
X_davinci_truthful_QA = df_davinci_truthful_QA.drop(labels="truthfulness", axis=1)

X_davinci_truthful_QA = X_davinci_truthful_QA.join(df_davinci_truthful_QA_original['question'])
X_davinci_truthful_QA = X_davinci_truthful_QA.join(df_davinci_truthful_QA_original['model_answer'])

X_train_davinci_truthful_QA, X_test_davinci_truthful_QA, y_train_davinci_truthful_QA, y_test_davinci_truthful_QA = (
    train_test_split(X_davinci_truthful_QA, y_davinci_truthful_QA, test_size=0.2, random_state=2023))

X_train_davinci_truthful_QA = X_train_davinci_truthful_QA.drop(labels="question", axis=1)
X_train_davinci_truthful_QA = X_train_davinci_truthful_QA.drop(labels="model_answer", axis=1)
question = X_test_davinci_truthful_QA['question']
model_answer = X_test_davinci_truthful_QA['model_answer']
X_test_davinci_truthful_QA = X_test_davinci_truthful_QA.drop(labels="question", axis=1)
X_test_davinci_truthful_QA = X_test_davinci_truthful_QA.drop(labels="model_answer", axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_davinci_truthful_QA = min_max_scaler.fit_transform(X_train_davinci_truthful_QA)
print("dumping scaler...")
pickle.dump(min_max_scaler, open("./data/scaler_2.pkl", "wb"))
print("dumped scaler")
min_max_scaler = preprocessing.MinMaxScaler()
X_test_davinci_truthful_QA = min_max_scaler.fit_transform(X_test_davinci_truthful_QA)  # why fit again on test data??

clf_davinci = svm.SVC(kernel='rbf', C=0.8)

# train ****************************************************************************************************************
clf_davinci.fit(X_train_davinci_truthful_QA, y_train_davinci_truthful_QA)
# pickle for later use
pickle.dump(clf_davinci, open('svm_model_davinci_2.pkl', 'wb'))
print("Dataset going into the predict method of the model___________________")
print("Shape: ", X_train_davinci_truthful_QA.shape)
# print(X_test_davinci_truthful_QA)
y_pred_test_davinci_truthful_QA = clf_davinci.predict(X_test_davinci_truthful_QA)
print("Predicted----------------------------------------------------------------")
print(y_pred_test_davinci_truthful_QA)
print("y true values of the test ----------------------------------------------------")
print(y_test_davinci_truthful_QA)
print("Accuracy-train-davincitruthfulQA-test-davincitruthfulQA-sfs:",metrics.accuracy_score(y_test_davinci_truthful_QA, y_pred_test_davinci_truthful_QA))
print(type(y_test_davinci_truthful_QA))
zeros = 0
ones = 0
tot = 0
for pred in y_train_davinci_truthful_QA:
    tot += 1
    if pred == 0:
        zeros += 1
    else:
        ones += 1
print("Zero class prior!!!!!!")
print(zeros/tot)
data = {
    'question': question.tolist(),
    'model_answer': model_answer.tolist(),
    'label': y_test_davinci_truthful_QA.tolist(),
    'pred': y_pred_test_davinci_truthful_QA.tolist()
}
df = pd.DataFrame(data)
df.to_csv('./outputs/pred_truthfulqa_davinci-sfs.csv', index=False)
