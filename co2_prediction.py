# وارد کردن کتابخانه‌های مورد نیاز
import pandas as pd  # برای کار با داده‌ها
from sklearn.model_selection import train_test_split  # برای تقسیم داده‌ها به آموزش و تست
from sklearn.linear_model import LinearRegression  # برای استفاده از رگرسیون خطی
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # برای ارزیابی مدل

# 1. بارگذاری داده‌ها
data = pd.read_csv('augmented_co2.csv')  # جایگذاری مسیر فایل داده‌ها
print("پیش‌نمایش داده‌ها:")
print(data.head())

# 2. جدا کردن ویژگی‌ها (X) و خروجی (y)
X = data[['engine', 'cylandr', 'fuelcomb']]  # متغیرهای ورودی
y = data['out1']  # متغیر خروجی

# 3. تقسیم‌بندی داده‌ها به دو مجموعه آموزشی و آزمایشی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# test_size=0.2 یعنی 20 درصد داده‌ها برای آزمایش و 80 درصد برای آموزش استفاده می‌شوند

# 4. ایجاد و آموزش مدل رگرسیون خطی
model = LinearRegression()  # ساخت مدل رگرسیون خطی
model.fit(X_train, y_train)  # آموزش مدل با داده‌های آموزشی

# 5. پیش‌بینی داده‌های آزمایشی
y_pred = model.predict(X_test)  # پیش‌بینی با داده‌های تست

# 6. ارزیابی عملکرد مدل
mae = mean_absolute_error(y_test, y_pred)  # میانگین خطای مطلق
mse = mean_squared_error(y_test, y_pred)  # میانگین مربع خطاها
r2 = r2_score(y_test, y_pred)  # امتیاز R2

# چاپ نتایج ارزیابی مدل
print("\nارزیابی مدل:")
print(f"میانگین خطای مطلق (MAE): {mae:.2f}")
print(f"میانگین مربع خطا (MSE): {mse:.2f}")
print(f"امتیاز R2: {r2:.2f}")

# 7. مشاهده ضرایب مدل (تأثیر هر ویژگی)
print("\nضرایب مدل (Weight for each feature):")
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coefficients)
