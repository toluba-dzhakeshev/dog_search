import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher.filters import Text
import numpy as np
import pandas as pd
import os
import csv
import faiss
import joblib
from sentence_transformers import SentenceTransformer, CrossEncoder

# Задаём токен Telegram-бота
API_TOKEN = '7536191497:AAEnh0z_znx9XEABbLnUeKNUv0NwfiN5StM'

# Включаем логирование
logging.basicConfig(level=logging.INFO)

# Инициализация бота, диспетчера и FSM (MemoryStorage для хранения состояний)
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

# Загрузка моделей и данных
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-base')

# Загрузка embedding-ов, faiss индекса и scaler'а
dog_embeddings = np.load('/Users/tolubai/Desktop/final_project/model/dog_embeddings.npy')
index = faiss.read_index('/Users/tolubai/Desktop/final_project/model/faiss_index.bin')
scaler = joblib.load('/Users/tolubai/Desktop/final_project/model/scaler.pkl')

# Загрузка данных о собаках (например, из CSV)
dog_data = pd.read_csv('/Users/tolubai/Desktop/final_project/datasets/cleaned_dataset.csv')
dog_data_copy = dog_data.copy()
original_dog_data = pd.read_csv('/Users/tolubai/Desktop/final_project/datasets/dog_breeds.csv')
dog_data = dog_data.to_dict(orient='records')

# Переменные для хранения данных пользователя
user_data = {}

min_price = dog_data_copy['Average Cost'].min()
max_price = dog_data_copy['Average Cost'].max()

min_height = dog_data_copy['Average Height'].min()
max_height = dog_data_copy['Average Height'].max()

min_weight = dog_data_copy['Average Weight'].min()
max_weight = dog_data_copy['Average Weight'].max()

# Определяем состояния (описание, цена, рост, вес, shedding, care_experience, количество)
class Form(StatesGroup):
    description = State()
    price = State()
    height = State()
    weight = State()
    shedding = State()
    care_experience = State()
    num_results = State()

# Обработчик команды /start с кнопкой 'Start'
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    # Создание кнопки "Начать"
    keyboard = InlineKeyboardMarkup()
    start_button = InlineKeyboardButton("Начать", callback_data="start")
    keyboard.add(start_button)
    await message.reply("Привет! Нажми 'Начать', чтобы подобрать собаку", reply_markup=keyboard)

# Обработчик нажатия на кнопку "Start"
@dp.callback_query_handler(Text(equals="start"))
async def process_start(callback_query: types.CallbackQuery):
    await Form.description.set()
    await bot.send_message(callback_query.from_user.id, "Опиши желаемую собаку:")

# Обработчик для ввода описания
@dp.message_handler(state=Form.description)
async def process_description(message: types.Message, state: FSMContext):
    user_data['description'] = message.text
    await Form.next()
    await message.reply("Укажи примерную цену (тыс. руб.):")

# Обработчик для ввода цены
@dp.message_handler(state=Form.price)
async def process_price(message: types.Message, state: FSMContext):
    try:
        price = float(message.text)
        if min_price <= price <= max_price:  # Проверка на диапазон цены из датасета
            user_data['price'] = price
            await Form.next()
            await message.reply(f"Укажи примерный рост собаки (см) в пределах от {min_height} до {max_height} см:")
        else:
            await message.reply(f"Пожалуйста, введи цену в пределах от {min_price} до {max_price} тыс. руб.")
    except ValueError:
        await message.reply("Пожалуйста, введи корректную цену (число).")

# Обработчик для ввода роста
@dp.message_handler(state=Form.height)
async def process_height(message: types.Message, state: FSMContext):
    try:
        height = float(message.text)
        if min_height <= height <= max_height:  # Проверка на диапазон роста из датасета
            user_data['height'] = height
            await Form.next()
            await message.reply(f"Укажи примерный вес собаки (кг) в пределах от {min_weight} до {max_weight} кг:")
        else:
            await message.reply(f"Пожалуйста, введи рост в пределах от {min_height} до {max_height} см.")
    except ValueError:
        await message.reply("Пожалуйста, введи корректный рост (число).")

# Обработчик для ввода веса
@dp.message_handler(state=Form.weight)
async def process_weight(message: types.Message, state: FSMContext):
    try:
        weight = float(message.text)
        if min_weight <= weight <= max_weight:  # Проверка на диапазон веса из датасета
            user_data['weight'] = weight
            await ask_shedding(message)
        else:
            await message.reply(f"Пожалуйста, введи вес в пределах от {min_weight} до {max_weight} кг.")
    except ValueError:
        await message.reply("Пожалуйста, введи корректный вес (число).")

# Вопрос о shedding
async def ask_shedding(message: types.Message):
    keyboard = InlineKeyboardMarkup(row_width=2)
    buttons = [
        InlineKeyboardButton("Мало линяет", callback_data="Sheds Little"),
        InlineKeyboardButton("Линяет", callback_data="Sheds"),
        InlineKeyboardButton("Много линяет", callback_data="Sheds A Lot"),
        InlineKeyboardButton("Не линяет", callback_data="No Sheds")
    ]
    keyboard.add(*buttons)
    
    await Form.shedding.set()  # Устанавливаем состояние shedding
    await bot.send_message(message.from_user.id, "Выберите тип линьки:", reply_markup=keyboard)

@dp.callback_query_handler(state=Form.shedding)
async def process_shedding(callback_query: types.CallbackQuery, state: FSMContext):
    user_data['sheds'] = callback_query.data
    await Form.next()  # Переходим к следующему вопросу
    await ask_care_experience(callback_query)

# Вопрос о Care Experience (да/нет)
async def ask_care_experience(callback_query: types.CallbackQuery):
    keyboard = InlineKeyboardMarkup(row_width=2)
    buttons = [
        InlineKeyboardButton("Да", callback_data="care_yes"),
        InlineKeyboardButton("Нет", callback_data="care_no")
    ]
    keyboard.add(*buttons)
    
    await Form.care_experience.set()  # Устанавливаем состояние
    await bot.send_message(callback_query.from_user.id, "Есть ли у тебя опыт ухода за собакой?", reply_markup=keyboard)

@dp.callback_query_handler(state=Form.care_experience)
async def process_care_experience(callback_query: types.CallbackQuery, state: FSMContext):
    user_data['care_experience'] = 1 if callback_query.data == "care_yes" else 0
    await Form.next()  # Переход к следующему состоянию
    await ask_num_results(callback_query)

# Вопрос о количестве собак
async def ask_num_results(callback_query: types.CallbackQuery):
    await bot.send_message(callback_query.from_user.id, "Сколько собак вывести?")

# Обработчик для ввода количества собак
@dp.message_handler(state=Form.num_results)
async def process_num_results(message: types.Message, state: FSMContext):
    if message.text.isdigit():
        user_data['num_results'] = int(message.text)
        await search_dogs(message)
        await state.finish()  # Завершаем состояние
        await ask_restart(message)  # Показываем кнопку для повторного старта
    else:
        await message.reply("Пожалуйста, введи корректное число.")
        
# Функция для показа кнопки "Начать сначала"
async def ask_restart(message: types.Message):
    keyboard = InlineKeyboardMarkup()
    restart_button = InlineKeyboardButton("Начать сначала", callback_data="start")
    keyboard.add(restart_button)
    await message.reply("Нажми 'Начать сначала', чтобы начать заново", reply_markup=keyboard)

# Функция для поиска собак (пример)
async def search_dogs(message: types.Message):
    # Здесь должна быть ваша логика поиска собак
    await message.reply("Собаки найдены!")

# Функция для обработки пользовательского запроса
def process_query(query, price, height, weight, sheds, care_experience, top_n):
    # Кодирование запроса (описание)
    query_embedding = sbert_model.encode(query, convert_to_tensor=True).cpu().numpy()

    # Нормализация числовых параметров (цена, рост, вес)
    query_numeric = np.array([price, height, weight]).reshape(1, -1)
    query_numeric = scaler.transform(query_numeric)

    # Фильтрация по shedding
    filtered_dog_data = dog_data_copy.copy()

    if sheds == 'Sheds Little':
        filtered_dog_data = filtered_dog_data[filtered_dog_data['Sheds Little'] == 1.0]
    elif sheds == 'Sheds':
        filtered_dog_data = filtered_dog_data[filtered_dog_data['Sheds'] == 1.0]
    elif sheds == 'Sheds A Lot':
        filtered_dog_data = filtered_dog_data[filtered_dog_data['Sheds A Lot'] == 1.0]
    elif sheds == 'No Sheds':
        # Здесь предполагается, что No Sheds означает, что ни один из трех параметров shedding не должен быть установлен.
        filtered_dog_data = filtered_dog_data[
            (filtered_dog_data['Sheds Little'] == 0.0) & 
            (filtered_dog_data['Sheds'] == 0.0) & 
            (filtered_dog_data['Sheds A Lot'] == 0.0)
        ]
        
    if care_experience == 0:  # Если опыт ухода не требуется
        filtered_dog_data = filtered_dog_data[filtered_dog_data['Care Experience'] == 0]
    elif care_experience == 1:  # Если опыт ухода требуется
        filtered_dog_data = filtered_dog_data[filtered_dog_data['Care Experience'] == 1]

    # Если нет собак, удовлетворяющих фильтру, возвращаем пустой список
    if filtered_dog_data.empty:
        return []

    results = []

    for i, row in filtered_dog_data.iterrows():
        # Комбинирование эмбеддингов, числовых данных и shedding для каждой собаки
        dog_sheds_data = row[['Sheds Little', 'Sheds', 'Sheds A Lot']].values
        dog_embedding = np.hstack((
            query_embedding.reshape(1, -1),  # Эмбеддинг текста
            query_numeric,                   # Числовые параметры
            dog_sheds_data.reshape(1, -1),   # Данные по shedding
            np.array([[care_experience]]).reshape(1, -1)  # Опыт ухода
        )).astype('float32')

        # Поиск ближайших соседей в FAISS индексе
        distances, indices = index.search(dog_embedding, top_n)

        # Формирование результатов
        results.append({
            "Name": row['Name'],
            "Description": row['Description'],
            "Relevance Score": distances[0][0]  # Используем только первый результат
        })

    return results[:top_n]

# Путь к CSV файлу
csv_file_path = '/Users/tolubai/Desktop/final_project/datasets/dog_search_results.csv'

# Функция для записи данных в CSV файл
def save_search_data(user_data, results):
    # Подготовка данных для записи
    rows = []
    for result in results:
        rows.append({
            'description': user_data['description'],
            'price': user_data['price'],
            'height': user_data['height'],
            'weight': user_data['weight'],
            'sheds': user_data['sheds'],
            'care_experience': user_data['care_experience'],
            'dog_name': result['Name'],
            'dog_description': result['Description'],
            'relevance_score': result['Relevance Score'],
            'final_score': result.get('Final Score', 'N/A')
        })

    # Проверяем, существует ли файл. Если нет — создаем его с заголовками
    if not os.path.exists(csv_file_path):
        df = pd.DataFrame(rows)
        df.to_csv(csv_file_path, mode='w', header=True, index=False)
    else:
        # Дописываем данные в конец существующего файла
        df = pd.DataFrame(rows)
        df.to_csv(csv_file_path, mode='a', header=False, index=False)

# Функция для поиска собак на основе введённых данных
async def search_dogs(message: types.Message):
    # Извлечение данных пользователя
    description = user_data['description']
    price = user_data['price']
    height = user_data['height']
    weight = user_data['weight']
    sheds = user_data['sheds']
    care_experience = user_data['care_experience']
    num_results = user_data['num_results']

    # Выполнение поиска
    results = process_query(description, price, height, weight, sheds, care_experience, num_results)

    # Весовые коэффициенты для критериев
    description_weight = 0.5
    price_weight = 0.3
    weight_weight = 0.2

    # Рассчитываем комбинированный скор для каждого результата
    for result in results:
        description_score = result['Relevance Score']  # Релевантность по описанию

        # Получаем породу из очищенного датасета
        breed_name = result['Name']
        breed_data = dog_data_copy[dog_data_copy['Name'] == breed_name].iloc[0]

        # Сравнение по цене и весу
        price_score = 1 - abs((breed_data['Average Cost'] - price) / price)
        weight_score = 1 - abs((breed_data['Average Weight'] - weight) / weight)

        # Итоговый комбинированный скор с учетом весов
        result['Final Score'] = (description_score * description_weight) + \
                                (price_score * price_weight) + \
                                (weight_score * weight_weight)

    # Сортировка по релевантности
    sorted_results = sorted(results, key=lambda x: x['Final Score'], reverse=True)
    
    save_search_data(user_data, sorted_results)

    # Формирование и отправка ответа
    for result in sorted_results:
        # Описание породы
        breed_name = result['Name']
        original_description = original_dog_data.loc[original_dog_data['Название породы'] == breed_name, 'Описание'].values[0]
        image_url = original_dog_data.loc[original_dog_data['Название породы'] == breed_name, 'Изображение'].values[0]
        cost = original_dog_data.loc[original_dog_data['Название породы'] == breed_name, 'Стоимость'].values[0]
        dog_link = original_dog_data.loc[original_dog_data['Название породы'] == breed_name, 'Ссылка'].values[0]

        # Ограничение до 50 слов
        description_words = original_description.split()
        limited_description = " ".join(description_words[:50]) + "..."

        # Подготовка сообщения с фотографией
        caption = (
            f"Порода: {breed_name}\n"
            f"Описание: {limited_description}\n"
            f"Стоимость: {cost}\n"
            f"Подробнее: {dog_link}"
        )
        
        # Создаем кнопки для каждой собаки
        keyboard = InlineKeyboardMarkup()
        button_yes = InlineKeyboardButton("Подошла", callback_data=f"yes_{breed_name}")
        button_no = InlineKeyboardButton("Не подошла", callback_data=f"no_{breed_name}")
        keyboard.add(button_yes, button_no)

        # Отправка фото с описанием
        if image_url:
            await bot.send_photo(message.chat.id, image_url, caption=caption, reply_markup=keyboard)
            
# Обработчик для кнопки "Подошла"
@dp.callback_query_handler(lambda c: c.data.startswith('yes_'))
async def process_yes_feedback(callback_query: types.CallbackQuery):
    breed_name = callback_query.data[4:]  # Получаем название породы из callback_data
    await bot.answer_callback_query(callback_query.id, text=f"Вы выбрали, что собака {breed_name} подошла.")

    # Записываем результат в файл
    save_feedback(callback_query.from_user.id, breed_name, 'Подошла')

# Обработчик для кнопки "Не подошла"
@dp.callback_query_handler(lambda c: c.data.startswith('no_'))
async def process_no_feedback(callback_query: types.CallbackQuery):
    breed_name = callback_query.data[3:]  # Получаем название породы из callback_data
    await bot.answer_callback_query(callback_query.id, text=f"Вы выбрали, что собака {breed_name} не подошла.")

    # Записываем результат в файл
    save_feedback(callback_query.from_user.id, breed_name, 'Не подошла')

# Функция для сохранения обратной связи
def save_feedback(user_id, breed_name, feedback):
    with open('/Users/tolubai/Desktop/final_project/datasets/feedback_dataset.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, breed_name, feedback])

# Запуск бота
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
