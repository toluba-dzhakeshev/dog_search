from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import csv
import time

# Настройки для chromedriver
service = Service('/Users/tolubai/Desktop/final_project/chromedriver')
options = Options()
options.add_argument("--disable-extensions")

# Создаем экземпляр драйвера
driver = webdriver.Chrome(service=service, options=options)

# URL страницы со списком пород
url = "https://doge.ru/poroda"

# Открываем страницу со списком пород
driver.get(url)

# Немного ждем, чтобы страница полностью загрузилась
time.sleep(5)

# Получаем HTML-код страницы и передаем его в BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'html.parser')

# Находим все ссылки на страницы с описанием пород
breed_links = soup.find_all('a', class_='breeds-letter-content__link')

# Создаем CSV файл для записи данных
with open('dog_breeds.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Название породы', 'Ссылка', 'Стоимость', 'Длительность жизни', 'Шерсть', 'Рождаемость', 'Рост в холке', 'Вес', 'Содержание', 'Назначение', 'Описание', 'Изображение'])

    for link in breed_links:
        breed_url = link['href']
        breed_name = link.text.strip()

        # Открываем страницу конкретной породы
        driver.get(breed_url)
        time.sleep(5)  # Ждем загрузки страницы

        # Получаем HTML-код страницы и передаем его в BeautifulSoup
        breed_soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Парсим описание BERET VSE OPISANIE S SAITA, KRATKOE OPISANIE TOLKO 2 PERVIH CLASSA
        description_elements = breed_soup.find_all('div', class_='single__desc single__desc--breed-story')
        description = " ".join([desc.text.strip() for desc in description_elements])

        # Парсим изображение
        image_element = breed_soup.find('div', class_='single-main__slider-item slick-slide slick-current slick-active')
        img_tag = image_element.find('img')
        image_url = img_tag['src']
        
        # Парсим информацию из блоков "breeds-info-card"
        info_cards = breed_soup.find_all('a', class_='breeds-info-card')
        breed_info = {
            'Стоимость': 'Не указано',
            'Длительность жизни': 'Не указано',
            'Шерсть': 'Не указано',
            'Рождаемость': 'Не указано',
            'Рост в холке, см': 'Не указано',
            'Вес, кг': 'Не указано',
            'Содержание': 'Не указано',
            'Назначение': 'Не указано'
        }
        
        for card in info_cards:
            name = card.find('div', class_='breeds-info-card__name').text.strip()
            value = card.find('div', class_='breeds-info-card__desc').text.strip()
            if name in breed_info:
                breed_info[name] = value

        # Записываем данные в CSV
        writer.writerow([
            breed_name,
            breed_url,
            breed_info['Стоимость'],
            breed_info['Длительность жизни'],
            breed_info['Шерсть'],
            breed_info['Рождаемость'],
            breed_info['Рост в холке, см'],
            breed_info['Вес, кг'],
            breed_info['Содержание'],
            breed_info['Назначение'],
            description,
            image_url
        ])

# Закрываем браузер
driver.quit()

print("Парсинг завершен, данные сохранены в 'dog_breeds.csv'")
