import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    print("Dataset loaded successfully from CSV.")
    # Map features to match assignment description
    df['age'] = df['Age']
    df['education_level'] = pd.cut(df['Education'], bins=[0,1.5,2.5,3.5,5], 
                                   labels=['High School', "Bachelor's", 'Master’s', 'PhD'])
    df['years_of_experience'] = df['TotalWorkingYears'].fillna(0)
    df['technical_test_score'] = df['PerformanceRating'] * 25  # Scale to 0-100
    df['interview_score'] = df['JobSatisfaction'] * 2.5  # Scale to 0-10
    df['previous_employment'] = np.where(df['NumCompaniesWorked'] > 0, 'Yes', 'No')
    df['suitable_for_employment'] = np.where(df['Attrition'] == 'No', 'Yes', 'No')
    # Select only relevant columns
    df = df[['age', 'education_level', 'years_of_experience', 'technical_test_score', 
             'interview_score', 'previous_employment', 'suitable_for_employment']].copy()
except FileNotFoundError:
    print("CSV not found. Using synthetic data for demo.")
    # Fallback: Generate synthetic data (as in previous example)
    np.random.seed(42)
    n = 1470
    age = np.clip(np.random.normal(36, 9, n).astype(int), 18, 65)
    education_map = {1: 'High School', 2: "Bachelor's", 3: 'Master’s', 4: 'PhD'}
    education_level_num = np.random.choice([1,2,3,4], n, p=[0.1, 0.5, 0.35, 0.05])
    education_level = [education_map[e] for e in education_level_num]
    years_of_experience = np.clip(np.random.poisson(7, n), 0, 40)
    technical_test_score = np.clip(np.random.normal(70, 15, n), 0, 100)
    interview_score = np.clip(np.random.normal(7, 1.5, n), 0, 10)
    previous_employment = np.random.choice(['Yes', 'No'], n, p=[0.85, 0.15])
    prob_suitable = 0.5 + 0.1*(age-36)/10 + 0.15*(technical_test_score-70)/30 + 0.2*(interview_score-7)/3 + \
                    0.1*(years_of_experience-7)/10 + 0.3*(education_level_num-2.5)/1.5 + 0.2*(previous_employment=='Yes')
    prob_suitable = np.clip(prob_suitable, 0, 1)
    suitable_for_employment = np.random.binomial(1, prob_suitable) 
    suitable_for_employment = np.where(suitable_for_employment == 1, 'Yes', 'No')
    df = pd.DataFrame({
        'age': age, 'education_level': education_level, 'years_of_experience': years_of_experience,
        'technical_test_score': technical_test_score, 'interview_score': interview_score,
        'previous_employment': previous_employment, 'suitable_for_employment': suitable_for_employment
    })

print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nNull values:")
print(df.isnull().sum())
print("\nDistribution of target:")
print(df['suitable_for_employment'].value_counts())

# Task 2: Data Preprocessing
# Encode categorical variables
le_edu = LabelEncoder()
le_prev = LabelEncoder()
le_target = LabelEncoder()

df['education_level_encoded'] = le_edu.fit_transform(df['education_level'])
df['previous_employment_encoded'] = le_prev.fit_transform(df['previous_employment'])
df['suitable_for_employment_encoded'] = le_target.fit_transform(df['suitable_for_employment'])

# Define features and target
features = ['age', 'education_level_encoded', 'years_of_experience', 'technical_test_score', 'interview_score', 'previous_employment_encoded']
X = df[features]
y = df['suitable_for_employment_encoded']

# Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Task 3: Model Building
clf = DecisionTreeClassifier(random_state=42, max_depth=5)  # Limit depth for simpler tree
clf.fit(X_train, y_train)
print("\nModel trained successfully.")

# Task 4: Model Visualization (Text-based; graphical saved as PNG)
tree_text = export_text(clf, feature_names=features)
print("\nDecision Tree Structure (text):")
print(tree_text)

# Graphical tree (saves to file)
plt.figure(figsize=(20,10))
from sklearn import tree
tree.plot_tree(clf, feature_names=features, class_names=le_target.classes_, filled=True)
plt.savefig('decision_tree.png')
plt.close()  # Close to avoid display in terminal
print("\nGraphical tree saved as 'decision_tree.png'.")

# Task 5: Model Testing and Prediction
y_pred = clf.predict(X_test)

hypotheticals = [
    {'age': 28, 'education_level': "Bachelor's", 'years_of_experience': 3, 'technical_test_score': 75, 'interview_score': 8, 'previous_employment': 'Yes'},
    {'age': 22, 'education_level': 'High School', 'years_of_experience': 0, 'technical_test_score': 55, 'interview_score': 5, 'previous_employment': 'No'},
    {'age': 45, 'education_level': 'Master’s', 'years_of_experience': 15, 'technical_test_score': 85, 'interview_score': 9, 'previous_employment': 'Yes'}
]

for i, cand in enumerate(hypotheticals, 1):
    cand_encoded = {
        'age': cand['age'],
        'education_level_encoded': le_edu.transform([cand['education_level']])[0],
        'years_of_experience': cand['years_of_experience'],
        'technical_test_score': cand['technical_test_score'],
        'interview_score': cand['interview_score'],
        'previous_employment_encoded': le_prev.transform([cand['previous_employment']])[0]
    }
    pred = clf.predict(pd.DataFrame([cand_encoded]))[0]
    pred_label = le_target.inverse_transform([pred])[0]
    print(f"\nHypothetical Candidate {i}: {cand}")
    print(f"Prediction: {pred_label} (Suitable: {'Yes' if pred_label == 'Yes' else 'No'})")

# Task 6: Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=le_target.classes_)

print("\nAccuracy Score:", round(accuracy, 4))
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le_target.classes_, yticklabels=le_target.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()
print("\nConfusion matrix plot saved as 'confusion_matrix.png'.")

importances = clf.feature_importances_
feature_imp = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_imp)

plt.figure(figsize=(10,6))
sns.barplot(data=feature_imp, x='importance', y='feature')
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
plt.close()
print("\nFeature importance plot saved as 'feature_importance.png'.")

