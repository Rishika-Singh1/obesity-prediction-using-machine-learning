import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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
