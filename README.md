🧠 Obesity Level Prediction Using Machine Learning
📌 Project Overview

Obesity has become a major health concern worldwide due to changing lifestyle habits and reduced physical activity. This project aims to predict obesity levels using machine learning techniques based on eating habits, physical condition, and daily lifestyle factors.

The project applies both supervised learning (for prediction) and unsupervised learning (for pattern discovery) to analyze obesity-related data and extract meaningful insights.

📂 Dataset Description

Dataset Source: UCI Machine Learning Repository

Total Records: 2111

Total Features: 17

Target Variable: NObeyesdad (Obesity Level)

The dataset includes information such as:

Age, Height, Weight

Eating habits

Physical activity levels

Lifestyle indicators

The dataset has a balanced class distribution, reducing bias and making it suitable for predictive modeling.

🎯 Objectives

To analyze obesity patterns using lifestyle and physical attributes

To build classification models for predicting obesity levels

To compare multiple machine learning algorithms

To identify the best-performing model

To discover hidden patterns using clustering techniques

⚙️ Methodology

The project was carried out in the following steps:

Data Loading & Exploration

Dataset inspection

Missing value check

Target variable distribution

Exploratory data analysis using visualizations

Data Preprocessing

One-hot encoding of categorical features

Feature scaling for distance-based models

Train-test split (80:20)

Supervised Learning (Classification)

Logistic Regression

K-Nearest Neighbors (KNN)

Naive Bayes

Decision Tree

Support Vector Machine (SVM)

Model Evaluation

Accuracy

Confusion Matrix

Unsupervised Learning (Clustering)

K-Means Clustering (Elbow Method)

Hierarchical Clustering (Dendrogram + Agglomerative)

🤖 Machine Learning Models Used
🔹 Logistic Regression

Applied for multiclass classification

Standard scaling improved convergence and stability

Accuracy: 0.87

🔹 K-Nearest Neighbors (KNN)

Tested for K values from 1 to 20

Best performance achieved at K = 1

Accuracy: 0.90

🔹 Naive Bayes

Lower performance due to feature dependency

Accuracy: 0.56

🔹 Decision Tree

Best-performing model

Handles nonlinear relationships effectively

Accuracy: 0.95

🔹 Support Vector Machine (SVM)

Lower accuracy due to multiclass and nonlinear data

Accuracy: 0.57

📊 Model Performance Summary
| Model               | Accuracy |
| ------------------- | -------- |
| Decision Tree       | **0.95** |
| KNN                 | 0.90     |
| Logistic Regression | 0.87     |
| SVM                 | 0.57     |
| Naive Bayes         | 0.56     |


🔍 Clustering Analysis
🔹 K-Means Clustering

Elbow Method used to select K = 4

Clusters showed strong alignment with obesity categories

Visualization done using Weight vs Age

🔹 Hierarchical Clustering

Dendrogram used to determine cluster structure

Agglomerative clustering applied with 4 clusters

Results consistent with K-Means, confirming stability

🧠 Key Insights

Decision Tree achieved the highest accuracy

Lifestyle and physical attributes strongly influence obesity levels

Clustering revealed natural groupings related to obesity categories

Consistent results across supervised and unsupervised methods validate the analysis

⚠️ Limitations

Dataset contains synthetic data

Hyperparameter tuning was limited to syllabus constraints

Advanced models like ensemble or deep learning were not used

Evaluation metrics were restricted to accuracy and confusion matrix

🚀 Future Scope

Use real-world medical or government health datasets

Apply ensemble models like Random Forest

Include additional evaluation metrics

Develop a health recommendation system or web application

🛠️ Tools & Technologies

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

📌 Conclusion

This project demonstrates the effectiveness of machine learning techniques in predicting obesity levels and uncovering meaningful lifestyle patterns. The results highlight the importance of data-driven approaches in health analysis and decision-making.

👩‍💻 Author

Diksha Tripathi
B.Tech CSE | Lovely Professional University