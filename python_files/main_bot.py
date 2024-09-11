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
from ultralytics import YOLO
from aiogram.dispatcher.filters import CommandStart, Text
import requests
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# Задаём токен Telegram-бота
API_TOKEN = os.getenv("API_TOKEN")

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

# Загрузка YOLO модели
MODEL_PATH = os.path.join("/Users/tolubai/Desktop/final_project/model", "weight.pt")
model = YOLO(MODEL_PATH).to('cpu')
dog_breeds_df = pd.read_csv('/Users/tolubai/Desktop/final_project/datasets/dogdf.csv')

#*******
def get_main_menu_keyboard():
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=False)
    main_menu_button = KeyboardButton("Вернуться в главное меню")
    keyboard.add(main_menu_button)
    return keyboard

@dp.message_handler(lambda message: message.text == "Вернуться в главное меню", state='*')
async def go_to_main_menu(message: types.Message, state: FSMContext):
    # Сбрасываем текущее состояние пользователя
    await state.finish()  # Используем finish для сброса всех состояний
    
    # Переводим пользователя в состояние choose_action
    await MainStates.choose_action.set()
    
    # Отправляем главное меню
    await send_welcome(message)
#***********

# Определение состояний
class MainStates(StatesGroup):
    choose_action = State()
    find_breed = State()
    describe_dog = State()

# Основное меню с выбором действия
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    user_name = message.from_user.first_name  # Получаем имя пользователя
    welcome_text = (
        f"Привет, {user_name}! 👋\n"
        "Я бот, который поможет тебе определить породу собаки по фото или подобрать собаку на основе твоего описания. "
        "Вот что я умею:\n"
        "- 📸 Отправь мне фото собаки, и я постараюсь определить её породу.\n"
        "- 📝 Опиши, какую собаку ты хочешь, и я подберу варианты.\n"
        "\nЧто бы ты хотел сделать?"
    )
    keyboard = InlineKeyboardMarkup(row_width=2)
    find_breed_button = InlineKeyboardButton("Определить породу по фото", callback_data="find_breed")
    describe_dog_button = InlineKeyboardButton("Подобрать собаку по запросу", callback_data="describe_dog")
    keyboard.add(find_breed_button, describe_dog_button)
    
    await MainStates.choose_action.set()
    await message.reply(welcome_text, reply_markup=keyboard)

# Обработчик выбора действия
@dp.callback_query_handler(Text(startswith="find_breed"), state=MainStates.choose_action)
async def choose_find_breed(callback_query: types.CallbackQuery, state: FSMContext):
    await bot.send_message(callback_query.from_user.id, "Отправь мне фото или ссылку на изображение, чтобы я определил её породу:")
    await MainStates.find_breed.set()
    await bot.answer_callback_query(callback_query.id)

@dp.callback_query_handler(Text(startswith="describe_dog"), state=MainStates.choose_action)
async def choose_describe_dog(callback_query: types.CallbackQuery, state: FSMContext):
    await bot.send_message(callback_query.from_user.id, "Опиши желаемую собаку:")
    await Form.description.set()
    await bot.answer_callback_query(callback_query.id)
    
#-----------------------------------------------------------------------------------------------------------------------------

class States(StatesGroup):
    start = State()
    photo = State()

# Логика для определения породы по фото (YOLO)
@dp.message_handler(state=MainStates.find_breed, content_types=['photo'])
async def photo_handler(message: types.Message, state: FSMContext):
    # Получаем фото из сообщения
    photo = message.photo[0]

    # Скачиваем фото
    file_id = photo.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    await bot.download_file(file_path, f"dog_{message.from_user.id}.jpg")

    # Обрабатываем изображение
    await handle_image_from_file(message, state, f"dog_{message.from_user.id}.jpg") 
    
    await state.update_data(last_action='find_breed')
    await state.reset_state(with_data=False)
    await ask_to_continue_or_return(message)
    
# Обработчик текстовых сообщений (ссылок)
@dp.message_handler(state=MainStates.find_breed, content_types=['text'])
async def text_handler(message: types.Message, state: FSMContext):
    # Проверяем, является ли сообщение ссылкой на изображение
    if message.text.startswith("http"):
        try:
            # Скачиваем изображение по ссылке
            response = requests.get(message.text, stream=True)
            response.raise_for_status()  # Проверяем статус ответа
            with open(f"dog_{message.from_user.id}.jpg", "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            await handle_image_from_file(message, state, f"dog_{message.from_user.id}.jpg") 
            
            await state.update_data(last_action='find_breed')
            await state.reset_state(with_data=False)
            await ask_to_continue_or_return(message)
            
        except requests.exceptions.RequestException as e:
            await message.answer(f"Ошибка при загрузке изображения: {e}")
    else:
        await message.reply("Пожалуйста, отправьте фото или ссылку на изображение.", reply_markup=get_main_menu_keyboard())

async def handle_image_from_file(message: types.Message, state: FSMContext, file_path: str):
    # Определяем породу
    results = model(file_path)

    # Проверяем, что YOLO определил породу
    if len(results[0].boxes.cls) > 0:
        # Извлекаем название породы и уверенность
        most_likely_class_index = int(results[0].boxes.cls[0])
        breed = results[0].names[most_likely_class_index]
        confidence = results[0].boxes.conf[0].item()  # Получаем уверенность

        # Получаем информацию о породе из DataFrame
        try:
            breed_info = dog_breeds_df[dog_breeds_df["label_name"] == breed]
            label = breed_info["label_ru"].iloc[0]
            description = breed_info["description"].iloc[0]
            link = breed_info["link"].iloc[0]

            # Отправляем ответ с уверенностью
            await message.answer(f"Это {label}! \n\nУверенность модели: {confidence:.2f}\n\n{description}\n\nПодробнее: {link}")

        except IndexError:
            await message.answer("Извини, информация об этой породе отсутствует.")

    else:
        await message.answer("Извини, я не могу определить породу этой собаки. Попробуйте другое изображение")

    # Удаляем скачанный файл
    os.remove(file_path)
    
#-----------------------------------------------------------------------------------------------------------------------------

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
            await message.reply(f"Укажи рост желаемой собаки (от {min_height} до {max_height} см):")
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
            await message.reply(f"Укажи вес желаемой собаки (от {min_weight} до {max_weight} кг):")
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
        await state.update_data(last_action='describe_dog')
        await state.reset_state(with_data=False)
        await ask_to_continue_or_return(message)  
    else:
        await message.reply("Пожалуйста, введи корректное число.")

#-----------------------------------------------------------------------------------------------------------------------------
async def ask_to_continue_or_return(message: types.Message):
    # Создаем кнопки для выбора дальнейшего действия
    keyboard = InlineKeyboardMarkup(row_width=2)
    continue_button = InlineKeyboardButton("Продолжить здесь", callback_data="continue_here")
    return_button = InlineKeyboardButton("Перейти в меню", callback_data="go_to_choice")
    keyboard.add(continue_button, return_button)
    
    await message.reply("Что ты хочешь сделать дальше?", reply_markup=keyboard)
    
# Обработчик нажатия кнопки "Продолжить здесь"
@dp.callback_query_handler(Text(startswith="continue_here"))
async def continue_here(callback_query: types.CallbackQuery, state: FSMContext):
    # Извлекаем последнее действие
    data = await state.get_data()
    last_action = data.get('last_action')
    
    if last_action == 'describe_dog':
        await callback_query.message.reply("Опиши желаемую собаку снова:")
        await Form.description.set()
    elif last_action == 'find_breed':
        await callback_query.message.reply("Отправь мне фото собаки снова:")
        await MainStates.find_breed.set()
    else:
        # Если что-то пошло не так, выводим сообщение для отладки
        await callback_query.message.reply("Ошибка: не могу продолжить, так как нет сохраненного действия.")
    
    await bot.answer_callback_query(callback_query.id)

# Обработчик нажатия кнопки "Перейти к выбору"
@dp.callback_query_handler(Text(startswith="go_to_choice"))
async def go_to_choice(callback_query: types.CallbackQuery, state: FSMContext):
    await state.reset_state(with_data=False)
    
    # Если пользователь хочет вернуться к выбору, показываем меню с выбором действия
    await send_welcome(callback_query.message)
    await bot.answer_callback_query(callback_query.id)
    
#-----------------------------------------------------------------------------------------------------------------------------

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
