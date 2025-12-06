import pandas as pd
import matplotlib.pyplot as plt
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
