# E-commerce Retail User Churn Prediction using XGBoost
This project aims to predict whether a customer will purchase a product based on their browsing and purchasing behavior. The model is built using K-Means clustering for customer segmentation and XGBoost for classification.

## 1. Project Overview
This project aims to predict customer churn in an e-commerce platform using machine learning techniques. The primary objective is to analyze customer behavior and identify patterns that indicate whether a user is likely to stop purchasing from the platform. The project utilizes K-Means for clustering and XGBoost for predictive modeling.

## 2. Dataset Description
The dataset consists of historical e-commerce transaction data, including user demographics, purchase history, browsing behavior, and engagement metrics. The key features include:
- **User ID**: Unique identifier for each user.
- **Session Duration**: Total time spent on the platform.
- **Purchase Frequency**: Number of purchases over a specific period.
- **Cart Abandonment Rate**: Percentage of sessions where a user added items to the cart but did not complete the purchase.
- **Customer Segment**: Clusters identified using K-Means.
- **Churn Label**: Target variable indicating whether a user churned (1) or remained active (0).

## 3. Data Preprocessing
Data preprocessing is a crucial step in the project to clean and prepare the dataset for analysis. The key steps include:
- **Handling Missing Values**: Filling or removing missing data points.
- **Outlier Detection**: Identifying and treating extreme values.
- **Feature Scaling**: Normalization or standardization of numerical features.
- **Categorical Encoding**: Converting categorical variables into numerical format using one-hot encoding or label encoding.

## 4. Exploratory Data Analysis (EDA)
EDA is performed to understand the distribution of data and uncover hidden patterns. This includes:
- **Visualizing Customer Distribution**: Identifying active vs. churned users.
- **Purchase Trends**: Understanding how purchase behavior changes over time.
- **Correlation Analysis**: Finding relationships between variables.
- **K-Means Clustering Analysis**: Grouping users based on their behavior patterns.

## 5. Feature Engineering
Feature engineering involves transforming raw data into meaningful features for model training. This includes:
- **Creating Aggregated Features**: Summarizing user behavior over different time windows.
- **Interaction Features**: Combining multiple features to create new ones.
- **Feature Selection**: Choosing the most relevant features to improve model performance.

## 6. Model Training & Evaluation
The model is trained using XGBoost, a powerful gradient boosting algorithm. The steps include:
- **Splitting Data**: Dividing the dataset into training and testing sets.
- **Training the Model**: Using XGBoost with optimized hyperparameters.
- **Evaluating Performance**: Using metrics such as:
  - Accuracy
  - Precision, Recall, and F1-score
  - ROC-AUC Curve

## 7. Deployment
The trained model is deployed using **Streamlit**, allowing users to input customer data and receive churn predictions. The deployment process includes:
- **Building a Web Interface**: Designing an interactive UI for input and output display.
- **Loading the Trained Model**: Using the saved model for real-time predictions.
- **Deploying on Heroku**: Hosting the application on a cloud platform.

## 8. Conclusion & Insights
Based on the analysis and model performance, the following insights were obtained:
- Customers with higher engagement and frequent purchases have lower churn rates.
- Abandoned cart rates and session duration are strong predictors of churn.
- The model achieves high predictive accuracy, helping businesses identify at-risk customers and take proactive measures.

This documentation provides a structured overview of the project, detailing each phase from data preprocessing to deployment. Let me know if any modifications are needed!

