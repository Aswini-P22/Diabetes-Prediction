
# NAME: ASWINI P

# AI/ML INTERNSHIP

# # DIABETES PREDICTION


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:\\Users\\Praveen\\Desktop\\diabetes.csv")

data.head()

data.tail()

data.describe()

data.info()

data.isna().sum()

data.duplicated().sum()


plt.figure(figsize = (10,6))
sns.countplot(x = 'Outcome', data=data)
plt.show()


from sklearn.preprocessing import StandardScaler
scc_x = StandardScaler()
X = pd.DataFrame(scc_x.fit_transform(data.drop(['Outcome'],axis = 1),), columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])

X.head()


y= data['Outcome']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)


from sklearn.neighbors import KNeighborsClassifier

test_scores =[]
train_scores = []
for i in range(1,15):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))


max_train_score = max(train_scores)
train_scores_index = [i for i, v in enumerate (train_scores) if v == max_train_score]
print("Max Train score {} % and k = {}".format(max_train_score*100, list(map(lambda x: x+1, train_scores_index))))

max_test_score = max(test_scores)
test_scores_index = [i for i, v in enumerate (test_scores) if v == max_test_score]
print("Max Test score {} % and k = {}".format(max_test_score*100, list(map(lambda x: x+1, test_scores_index))))

plt.figure(figsize = (12,5))
p= sns.lineplot(range(1,15), train_scores, marker = '*', label = 'Train Score')
p= sns.lineplot(range(1,15), test_scores, marker = 'o', label = 'Test Score')

knnn =KNeighborsClassifier(13)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)

from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# # Making Prediction

def predict_diabetes(input_data):
    input_data_np = np.asarray(input_data)
    input_data_reshaped = input_data_np.reshape(1,-1)
    prediction = knn.predict(input_data_reshaped)
    if prediction[0] == 1:
        return 'Diabetic'
    else:
        return 'Not Diabetic'

user_input = [2,90,74,0.3,17,18.7,0,18]
result = predict_diabetes(user_input)
print("The Person is ",result)

