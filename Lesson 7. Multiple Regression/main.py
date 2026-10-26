from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

print("Привіт! Працюємо з множина регресією")

# Отримати набір даних із Каліфорнія житло - 1990
# as_frame - дані у вигляді DataFrmae
housing = fetch_california_housing(as_frame=True)

#Отримало сам DataFrame
df = housing.frame

# print(df.head()) # Вивести 5 перших рядків із даного dataset

# Обираємо ознаки для множиної регресії
df = df[[
    "MedInc", # Середній дохід населення району
    "HouseAge", # Середній вік будинку в районі
    "AveRooms", # Середня кількість кімнат в районі
    "MedHouseVal" # Середня ціна будинку - цільова ознака
]]

# Нова колонка - категоріальний поділ
df["Category"] = np.where(df["HouseAge"] > 30, "Old", "New")

print("Нова таблиця для множинної регресії")
# print(df.head())

# Виділяємо ознаки по X та Y
X = df.drop(columns="MedHouseVal")
# print(X.head())

y = df["MedHouseVal"]
# print(y.head())

# Розподіл даних і налаштовуємо random
# X - Список ознак по яких будемо робити аналіз - експертизу
# y - Значення, яке потрібно прогозувати - цільова ознака
# test_size - 20% це будуть дані для тесту
# 80% - це для навчання - виявлення закономірносте даних
# random_state=42 - для того, щоб дані завжди були однаковими
X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.2, random_state=42))

# проводимо обробку даних
numeric_features = [
    "MedInc",
    "HouseAge",
    "AveRooms",
]

categorical_features = [
    "Category"
]

preprocessor = ColumnTransformer(
    #Правила для обробки інформації
    transformers=[
        (
            # Назва трансформації
            "num",
            # Буде усі числові конки приводити до значення - близького до 1
            # Віднімаємо середнє і ділимо на стандартне відхилення
            StandardScaler(),
            # Це будемо робити на основі ось цих даних
            numeric_features,
        ),
        (
            "cat",
            # воно приводити дані до числоваого формату
            # при цьому буде лише 1 колонка у які буде вказано 0 або 1
            OneHotEncoder(drop="first"),
            categorical_features, # Застосовуємо до цих даних
        )
    ]
)

# Будуємо модель для роботи із ознаками
model = Pipeline([
    (
# назва 1 етапу
        "preprocessor",
# Попередня обробка даних: Стандартизація та Кодування Категоріальних ознак
        preprocessor
    ),
    (
        "regressor", # Назва 2 етапу
        LinearRegression() # Множина лінійна регресія
    )
])

# Тепер модель знає з якими даним її працювати
model.fit(X_train, y_train) # проходить машине навчання

# Отримаємо множину пронозів для тестових дани
y_pred = model.predict(X_test)

print(y_pred)

#Оцінка моделі
r2 = r2_score(y_test, y_pred) # порівняємо, що вийшло з тим, що малов вийти

# шукаємо абсолютну похибку
mea = mean_absolute_error(y_test, y_pred)

#Середньо квадратична похибка даних
rmse = root_mean_squared_error(y_test, y_pred)

print("r2 score:", r2)
print("mea score:", mea) # похибка очислень - 0.6 - якщо будинок коштує сотні тисяч - то похибка 60 тисяч
print("rmse score:", rmse) # Дана змінна реагує на досить великі відхилення від норми

# ==========================================
# 11. Коефіцієнти моделі
# ==========================================
# Формуємо список назв усіх ознак,
# які були використані моделлю після попередньої обробки.
feature_names = (
    # Додаємо числові ознаки.
    numeric_features +
    # Додаємо назви нових стовпців,
    # створених OneHotEncoder.
    list(
        # Отримуємо об'єкт попередньої обробки (ColumnTransformer)
        # із Pipeline.
        model.named_steps["preprocessor"]
                # Отримуємо трансформатор категоріальних ознак.
             .named_transformers_["cat"]
# Повертаємо назви стовпців після OneHotEncoder.
             # Наприклад:
             # Category → Category_New
             .get_feature_names_out(categorical_features)
    )
)
# Отримуємо коефіцієнти навченої моделі
# множинної лінійної регресії.
coef = model.named_steps["regressor"].coef_

# Створюємо DataFrame,
# у якому кожній ознаці відповідає її коефіцієнт.
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coef
})
# Виводимо заголовок.
print("\nКоефіцієнти моделі:")
# Виводимо таблицю коефіцієнтів.
print(coef_df)