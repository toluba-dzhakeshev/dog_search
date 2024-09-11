import asyncio
import logging
import os
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import CommandStart, Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ContentType, Message, PhotoSize
from ultralytics import YOLO
import pandas as pd

# Настройка логгирования
logging.basicConfig(level=logging.INFO)

# Токен бота
API_TOKEN = '7173102633:AAG_GyzGVowFPoggHgJPhGTgcB2Hp-WUyBM'

# Путь к модели YOLO
MODEL_PATH = os.path.join("/Users/tolubai/Desktop/final_project/model", "weight.pt")

# Путь к файлу с описанием пород
DOG_BREEDS_FILE = os.path.join("/Users/tolubai/Desktop/final_project/datasets/dogdf.csv")

# Создаем объект бота
bot = Bot(token=API_TOKEN)

# Создаем диспетчер для обработки событий
dp = Dispatcher(bot, storage=MemoryStorage())

# Загружаем модель YOLO
model = YOLO(MODEL_PATH)

# Загружаем таблицу с описанием пород
dog_breeds_df = pd.read_csv(DOG_BREEDS_FILE)

class States(StatesGroup):
    start = State()
    photo = State()

# Обработчик команды /start
@dp.message_handler(CommandStart())
async def send_welcome(message: types.Message, state: FSMContext):
    user_name = message.from_user.first_name
    await message.answer(f"Привет, {user_name}! Отправьте мне фото собаки, чтобы узнать ее породу, или ссылку на изображение.")
    await state.set_state(States.start)

# Обработчик фотографий
@dp.message_handler(state=States.start, content_types=['photo'])
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

# Обработчик текстовых сообщений (ссылок)
@dp.message_handler(state=States.start, content_types=['text'])
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
        except requests.exceptions.RequestException as e:
            await message.answer(f"Ошибка при загрузке изображения: {e}")
    else:
        await message.answer("Пожалуйста, отправьте фото или ссылку на изображение.")

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

if __name__ == "__main__":
    asyncio.run(dp.start_polling())