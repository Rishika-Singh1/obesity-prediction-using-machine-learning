# 🧠 Obesity Level Prediction Using Machine Learning

## 📌 Project Overview
Obesity has become a major health concern due to unhealthy eating habits and reduced physical activity. This project focuses on predicting obesity levels using machine learning techniques based on eating habits, physical condition, and lifestyle attributes. Both supervised and unsupervised learning approaches are used to analyze the data and extract meaningful insights.

---

## 📂 Dataset Description
- **Source:** UCI Machine Learning Repository  
- **Total Records:** 2111  
- **Total Features:** 17  
- **Target Variable:** `NObeyesdad` (Obesity Level)

The dataset contains attributes such as age, height, weight, eating habits, physical activity, and lifestyle factors. The class distribution is balanced, making the dataset suitable for predictive modeling with minimal bias.

---

## 🎯 Objectives
- Analyze obesity patterns using lifestyle and physical attributes  
- Build machine learning models for obesity level prediction  
- Compare the performance of multiple classification algorithms  
- Identify the best-performing model  
- Discover hidden patterns using clustering techniques  

---

## ⚙️ Methodology
1. **Data Exploration**
   - Dataset inspection and visualization  
   - Missing value analysis  
   - Target variable distribution  

2. **Data Preprocessing**
   - One-hot encoding of categorical features  
   - Feature scaling for distance-based algorithms  
   - Train-test split (80% training, 20% testing)  

3. **Supervised Learning**
   - Logistic Regression  
   - K-Nearest Neighbors (KNN)  
   - Naive Bayes  
   - Decision Tree  
   - Support Vector Machine (SVM)  

4. **Model Evaluation**
   - Accuracy  
   - Confusion Matrix  

5. **Unsupervised Learning**
   - K-Means Clustering (Elbow Method)  
   - Hierarchical Clustering (Dendrogram and Agglomerative approach)  

---

## 🤖 Machine Learning Models Used
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Decision Tree  
- Support Vector Machine (SVM)  

---

## 📊 Model Performance Summary

| Model | Accuracy |
|------|----------|
| Decision Tree | **0.95** |
| KNN | 0.90 |
| Logistic Regression | 0.87 |
| SVM | 0.57 |
| Naive Bayes | 0.56 |

---

## 🔍 Clustering Analysis

### 🔹 K-Means Clustering
- Elbow Method used to determine optimal number of clusters  
- Best value of **K = 4**  
- Clusters showed strong alignment with obesity categories  

### 🔹 Hierarchical Clustering
- Dendrogram used to analyze cluster structure  
- Agglomerative clustering applied with four clusters  
- Results were consistent with K-Means clustering  

---

## 🧠 Key Insights
- Decision Tree achieved the highest accuracy  
- Lifestyle and physical attributes strongly influence obesity levels  
- Clustering techniques revealed meaningful natural groupings  
- Consistent patterns were observed across different models  

---

## ⚠️ Limitations
- Dataset contains synthetic data  
- Advanced hyperparameter tuning was not performed  
- Ensemble and deep learning models were not included  
- Evaluation metrics were limited to accuracy and confusion matrix  

---

## 🚀 Future Scope
- Use real-world healthcare datasets  
- Apply ensemble models such as Random Forest  
- Include additional evaluation metrics  
- Develop a health recommendation or awareness system  

---

## 🛠️ Tools and Technologies
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

---

## 📌 Conclusion
This project demonstrates how machine learning techniques can effectively predict obesity levels and uncover meaningful lifestyle patterns. The results highlight the importance of data-driven approaches in health analysis and decision-making.

---

## 👩‍💻 Author
**Rishika Singh**  
B.Tech - Computer Science and Engineering  
Lovely Professional University
