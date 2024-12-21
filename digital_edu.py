import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv('train.csv')

data.drop(columns=['life_main', 'people_main', 'career_start', 'career_end', 'id', 'last_seen'], inplace=True)

print(data.info())

print(data['bdate'].head())
# Функція для розбиття дати народження на рік

def split_bdate(bdate):
    if not isinstance(bdate, str):  # Перевірка, чи є bdate рядком
        return None
    parts = bdate.split('.')
    if len(parts) == 3:
        return int(parts[2])  # Повернення року
    elif len(parts) == 2:
        return None  # Відсутній рік

data['birth_year'] = data['bdate'].apply(split_bdate)

# Заповнення відсутніх років на основі медіани по кожній статі
def fill_byear(row):
    if pd.isnull(row['birth_year']):
        if row['sex'] == 1:
            return data[data['sex'] == 1]['birth_year'].median()
        else:
            return data[data['sex'] == 2]['birth_year'].median()
    return row['birth_year']

data['birth_year'] = data.apply(fill_byear, axis=1)

# Видалення оригінальної колонки 'bdate'
data.drop('bdate', axis=1, inplace=True)

print(data['birth_year'].head())
print(data.info())

# Функція для перетворення статі у числовий формат
def convert_sex(sex):
    if sex == 2:
        return 1
    else:
        return 0

data['sex'] = data['sex'].apply(convert_sex)

# Функція для перетворення форми навчання у числовий формат
def convert_education_form(education_form):
    if education_form == 'Full-time':
        return 1
    else:
        return 0

data['education_form'] = data['education_form'].apply(convert_education_form)

# Функція для перевірки наявності української мови у списку мов
def convert_langs(langs):
    if 'Українська' in str(langs):
        return 1
    else:
        return 0
    
data['langs'] = data['langs'].apply(convert_langs)

print(data['has_mobile'].value_counts())

data = pd.get_dummies(data, columns=['education_status', 'relation', 'occupation_type'], drop_first=True)

data.fillna(0, inplace=True)

print(data.info())

# Поділ на тренінговий та тестовий набори
x = data.drop('result', axis=1)
y = data['result']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Масштабування даних
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Побудова моделі k-найближчих сусідів
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# Прогнозування та оцінка моделі
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')


