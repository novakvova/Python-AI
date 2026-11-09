from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn # Модулі для створення нейронної мережі
import torch.optim as optim # Оптимізатори для навчання мережі

# Для робота dataset
iris = load_iris()

# print("Iris data loaded", iris.data.shape)
# print("Iris data loaded", iris.data[:5])
X = iris.data
# 1. sepal length — довжина чашолистка
# 2. sepal width  — ширина чашолистка
# 3. petal length — довжина пелюстки
# 4. petal width  — ширина пелюстки

y = iris.target
# print(X.shape, y.shape)
# print("y", y)
print("Характеристики")
for feature in iris.feature_names:
    print(" - ", feature)

print("\nКласи:")
for class_name in iris.target_names:
    print(" - ", class_name)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, #20% для тесту
    random_state=42,
    stratify=y # зберігати позицію даних
    )

# print("Навчальні приклади: ",len(X_train))
# print("Тестові приклади: ",len(X_test))

# print("X_train:", X_test[:5])
# print("X_test:", X_test[:5])

#Маштабування даних
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # змінуюємо дані для зручного маштабування в мережі
X_test = scaler.transform(X_test)

# print("X_train:", X_train[:5])
# print("X_test:", X_test[:5])

#Перетворюємо дані у тензор
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Робимо багатошаровий персептрон MLP
class IrisMLP(nn.Module):
    def __init__(self):
        super(IrisMLP, self).__init__()

        #Створюємо шари для мережі
        self.model = nn.Sequential(
            # із 4 ознак робимо 32 ознаки
            nn.Linear(4, 32),
            # Використовуємо функцію RELU щоб забарти нелінійність даних
            nn.ReLU(),
            # Робимо прихований шар
            # з 32 -> 8
            nn.Linear(32, 8),
            nn.ReLU(),
            # Завершуємо роботу і робимо вихідний шар
            # У ході роботи нейроної мережі у нас має вийти
            # Опис 3 класів
            # -  setosa            - versicolor            - virginica
            nn.Linear(8, 3)
        )
    # Пряме поширення сигналу у мережі
    # для кожного x - виконуємо усі алгоритми MLP
    def forward(self, x):
        return self.model(x)

# Робимо об'єкт для MLP
model = IrisMLP()
print("Архітектура MLP")
print(model)

# Будуємо фукнцію втрат
# Для того, щоб визначити наскільки прогноз відрізняється від правильної відповіді
criterion = nn.CrossEntropyLoss() # Тут ми виконуємо крос ентропію
# Налаштовуємо оптимізатор
# Використовується для зміни Ваги нейронної мережі
# Щоб на наступному етапі можна було врахувати кращу вагу і був кращий результат
optimizer = optim.Adam(model.parameters(), # параметри моделі
                       lr=0.01) #швидкість навчання
# використовуємо параметри навчання
epochs = 100 # Кількість епох

# чим менший loss - тим краще модель навчається
loss_history = [] # для виводу результату на кожній із епох

# На скільки відсотків правильний прогноз
acc_history = []

for epoch in range(epochs):
    model.train() # Тренуємо нашу модельі

    outputs = model(X_train_tensor) # навчені дані передаємо у нейрону мережі

    loss = criterion(outputs, # порівнюємо передбачення із правильним відповідями
                     y_train_tensor)

    optimizer.zero_grad() # Видаляє градієнт попередньої операції
    loss.backward() # повертаємо назад по мережі у пошуках ваги для кращого прогнозування
    # На оснвоі ваг - робить оптимізацію
    optimizer.step()

    # Викуємо найбільше значення серед рядків і робимо 1 масив із цими значеннями
    predictions = torch.argmax(outputs,1)

    # Порівнняння із правильним значенням
    correct = (predictions == y_train_tensor)
    acc = correct.float().mean() # Знаходимо середнє значення правильних відповідей

    loss_history.append(loss.item()) # для візуалізації
    acc_history.append(acc.item())  # для графіку

    if epoch % 20 == 0:
        print(f" - [{epoch+1:3}/{epochs}]")
        print(f" - Loss: {loss.item():.4f}")
        print(f" - Accuracy: {acc.item()*100:.2f}")


model.eval() # переводимо модель у режим очікування

# Відключаємо обчислення градієнтів
with torch.no_grad():

    # Передаємо тестові дані через мережу - Передати тестові дані через навчену нейронну мережу та отримати її прогнози.
    test_outputs = model(
        X_test_tensor
    )


    # Вибираємо клас з найбільшим значенням
    test_predictions = torch.argmax(
        test_outputs,
        dim=1
    )

# Перетворюємо Tensor у NumPy
predictions_numpy = test_predictions.numpy()


# Обчислюємо Accuracy
test_accuracy = accuracy_score(
    y_test,
    predictions_numpy
)

# Виводимо результат
print("\n================================")

# Виводимо точність
print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

print("================================")

# Створюємо звіт класифікації
report = classification_report(
    y_test,                          # Правильні класи
    predictions_numpy,               # Передбачені класи
    target_names=iris.target_names   # Назви класів
)


# Виводимо звіт
print("\nClassification Report:")
# precision - Серед усіх квіток, які модель назвала певним класом, скільки дійсно належать до цього класу?
# recall - Скільки реальних об'єктів певного класу модель змогла знайти?
# F1-score — це показник, який одночасно враховує Precision та Recall.
print(report)

# Обчислюємо матрицю помилок - покаже, які класи шутає між собою модель
cm = confusion_matrix(
    y_test,
    predictions_numpy
)


# Виводимо матрицю в консоль
print("\nConfusion Matrix:")
print(cm)

# Візуалізація матриці
# Створюємо область для графіка
plt.figure(
    figsize=(7, 6)
)


# Відображаємо матрицю як зображення
plt.imshow(
    cm,
    interpolation="nearest"
)


# Додаємо заголовок
plt.title(
    "Confusion Matrix — Iris MLP"
)


# Додаємо назву осі X
plt.xlabel(
    "Передбачений клас"
)


# Додаємо назву осі Y
plt.ylabel(
    "Справжній клас"
)


# Створюємо список позицій класів
classes = [
    "setosa",
    "versicolor",
    "virginica"
]


# Встановлюємо підписи осі X
plt.xticks(
    range(3),
    classes
)


# Встановлюємо підписи осі Y
plt.yticks(
    range(3),
    classes
)


# Додаємо числові значення всередину матриці
for i in range(3):
    # Перебираємо стовпці
    for j in range(3):
        # Виводимо значення конкретної клітинки
        plt.text(
            j,                    # Позиція X
            i,                    # Позиція Y
            cm[i, j],             # Значення
            ha="center",          # Горизонтальне вирівнювання
            va="center"           # Вертикальне вирівнювання
        )
# Автоматично налаштовуємо відступи
plt.tight_layout()


# Показуємо графік
plt.show()

# Створюємо новий графік
plt.figure(
    figsize=(8, 5)
)


# Малюємо зміну Loss
plt.plot(
    loss_history
)


# Назва осі X
plt.xlabel(
    "Epoch"
)


# Назва осі Y
plt.ylabel(
    "Loss"
)


# Заголовок графіка
plt.title(
    "Зміна помилки під час навчання"
)


# Вмикаємо сітку
plt.grid()


# Показуємо графік
plt.show()

# Створюємо новий графік
plt.figure(
    figsize=(8, 5)
)


# Малюємо зміну Accuracy
plt.plot(
    acc_history
)


# Назва осі X
plt.xlabel(
    "Epoch"
)


# Назва осі Y
plt.ylabel(
    "Accuracy"
)


# Заголовок
plt.title(
    "Точність MLP під час навчання"
)


# Вмикаємо сітку
plt.grid()


# Показуємо графік
plt.show()


# Створюємо характеристики нової квітки
sample = [[
    5.1,                              # Довжина чашолистка
    3.5,                              # Ширина чашолистка
    1.4,                              # Довжина пелюстки
    0.2                               # Ширина пелюстки
]]


# Масштабуємо нові дані
# використовуючи scaler, навчений на TRAIN
sample_scaled = scaler.transform(
    sample
)


# Перетворюємо нові дані у Tensor
sample_tensor = torch.tensor(

    sample_scaled,

    dtype=torch.float32
)


# Переводимо модель у режим оцінювання
model.eval()


# Відключаємо градієнти
with torch.no_grad():

    # Передаємо нову квітку в MLP
    output = model(
        sample_tensor
    )


    # Знаходимо індекс найбільш імовірного класу
    prediction = torch.argmax(
        output,
        dim=1
    ).item()

# Виводимо роздільник
print("\n================================")


# Виводимо характеристики квітки
print("Характеристики нової квітки:")
print(sample)

# Виводимо передбачений клас
print("\nПередбачений клас:")

# Отримуємо назву класу за його індексом
print(
    iris.target_names[prediction]
)

# Виводимо роздільник
print("================================")