
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# 1. טעינת הנתונים
df = pd.read_csv('parkinsons.csv')

# 2. בחירת מאפיינים (Features) ופלט (Output)
# ע"פ המאמר PPE ו-DFA הם מדדים חזקים
features = ['PPE', 'DFA']
output = 'status'

X = df[features]
y = df[output]

print("--- הצגת הנתונים הגולמיים ---")
print(X.head())
print(y.head())

# 3. נרמול הנתונים (Scaling)
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("\n--- הצגת הנתונים המנורמלים ---")
print(X_scaled.head())

# 4. חלוקה לסט אימון וסט בדיקה (Training / Validation)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\n--- בדיקת גודל הסטים ---")
print("Training set:", X_train.shape)
print("Validation set:", X_val.shape)

# 5. בחירת המודל (SVM עם גרעין RBF)
model = SVC(kernel='rbf')

# 6. אימון המודל ובדיקת דיוק
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# בדיקה אם הדיוק עומד בדרישה
if accuracy >= 0.8:
    print("Success: Accuracy is 0.8 or higher.")
else:
    print("Accuracy is below 0.8.")

# 7. שמירת המודל לקובץ
joblib.dump(model, 'parkinson_model.joblib')
print("Model saved as 'parkinson_model.joblib'")
