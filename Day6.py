import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("C:\\Users\\diksh\\Downloads\\Rishika ML Project\\ObesityDataSet_raw_and_data_sinthetic.csv")

print("Shape of dataset:", df.shape)
df.head()
df.info()
df.describe()

# Checking for missing values
df.isnull().sum()

# Checking target variable distribution
print(df['NObeyesdad'].value_counts())

# Age distribution
plt.hist(df['Age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

# Obesity level counts
df['NObeyesdad'].value_counts().plot(kind='bar')
plt.xlabel('Obesity Level')
plt.ylabel('Count')
plt.title('Obesity Level Distribution')
plt.xticks(rotation=45)
plt.show()

# Relationship between Weight and Height
plt.scatter(df['Height'], df['Weight'])
plt.xlabel('Height (m)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight')
plt.show()

# Correlation Heatmap (Using Seaborn)
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# Separate independent features and target
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Detecting Outliers using Boxplots
X.select_dtypes(include=['int64','float64']).plot(kind='box', figsize=(12,6))
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training Set:", X_train.shape, "Testing Set:", X_test.shape)

# LOGISTIC REGRESSION
# Predicting Obesity Level (NObeyesdad)
print("\n================ LOGISTIC REGRESSION =================")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)

y_pred_log = log_reg.predict(X_test_scaled)

acc_log = accuracy_score(y_test, y_pred_log)
cm_log = confusion_matrix(y_test, y_pred_log)

print("Accuracy (Logistic Regression):", round(acc_log, 2))
print("Confusion Matrix (Logistic Regression):\n", cm_log)

# Confusion Matrix Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm_log, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# K-NEAREST NEIGHBORS (KNN)
print("\n================ K-NEAREST NEIGHBORS (KNN) =================")

# Finding the best K value
accuracy_scores = []
k_values = range(1, 20)

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    y_pred_temp = knn_temp.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred_temp))

# Plot accuracy vs k
plt.figure(figsize=(10,5))
plt.plot(k_values, accuracy_scores, marker='o')
plt.title("KNN Accuracy vs K Value")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Training final KNN Model with best K
best_k = k_values[accuracy_scores.index(max(accuracy_scores))]
print(f"Best K Value: {best_k}")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

# Accuracy
acc_knn = accuracy_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)

print("Accuracy (KNN):", round(acc_knn, 2))
print("Confusion Matrix (KNN):\n", cm_knn)

# Confusion Matrix Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm_knn, annot=True, cmap='Greens', fmt='g')
plt.title(f'Confusion Matrix - KNN (K={best_k})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# NAIVE BAYES CLASSIFIER
print("\n================ NAIVE BAYES CLASSIFIER =================")

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

acc_nb = accuracy_score(y_test, y_pred_nb)
cm_nb = confusion_matrix(y_test, y_pred_nb)

print("Accuracy (Naive Bayes):", round(acc_nb, 4))
print("Confusion Matrix (Naive Bayes):\n", cm_nb)

# Confusion Matrix Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm_nb, annot=True, cmap='OrRd', fmt='g')
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# DECISION TREE & SVM
print("\n================ DECISION TREE CLASSIFIER =================")

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)

print("Accuracy (Decision Tree):", round(acc_dt, 2))
print("Confusion Matrix (Decision Tree):\n", cm_dt)

# Confusion Matrix Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm_dt, annot=True, cmap='Purples', fmt='g')
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\n================ SUPPORT VECTOR MACHINE (SVM) =================")

svm_clf = SVC()
svm_clf.fit(X_train, y_train)

y_pred_svm = svm_clf.predict(X_test)

acc_svm = accuracy_score(y_test, y_pred_svm)
cm_svm = confusion_matrix(y_test, y_pred_svm)

print("Accuracy (SVM):", round(acc_svm, 2))
print("Confusion Matrix (SVM):\n", cm_svm)

# Confusion Matrix Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm_svm, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# MODEL ACCURACY COMPARISON
model_names = ['Logistic Regression', 'KNN', 'Naive Bayes', 'Decision Tree', 'SVM']
accuracies = [acc_log, acc_knn, acc_nb, acc_dt, acc_svm]

plt.figure(figsize=(10,5))
sns.barplot(x=model_names, y=accuracies, hue=model_names, palette='viridis', legend=False)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.xticks(rotation=30)
plt.show()

print("\n================ FINAL MODEL PERFORMANCE SUMMARY ================")
for name, acc in zip(model_names, accuracies):
    print(f"{name}: {round(acc,2)}")
