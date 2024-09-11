# Используем официальный Python-образ в качестве базового
FROM python:3.10

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY txt_files/requirements.txt /app/requirements.txt

# Обновляем setuptools и устанавливаем зависимости
RUN pip3 install --upgrade setuptools
RUN pip3 install --no-cache-dir -r requirements.txt

# Копируем остальные файлы проекта
COPY . /app

# Открываем доступ для исполнения скриптов
RUN chmod 755 .

# Запускаем основного бота
CMD ["python", "./python_files/main_bot.py"]