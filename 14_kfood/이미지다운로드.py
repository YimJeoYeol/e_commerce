import os
from time import sleep
import urllib

from selenium import webdriver
import chromedriver_autoinstaller
from selenium.common import ElementNotInteractableException
from selenium.webdriver.common.by import By

검색어 = "제육볶음"  # 된장찌개, 돌솥비빔밥, 김치, 치킨, 삼겹살
try:
    os.mkdir(검색어)
except FileExistsError as e:
    pass

chromedriver_autoinstaller.install()
driver = webdriver.Chrome()

driver.get("https://www.google.com")

검색입력창 = driver.find_element(By.CSS_SELECTOR, "textarea[title='검색']")
검색입력창.send_keys(검색어)
검색입력창.submit()

이미지버튼 = driver.find_element(By.XPATH, "/html/body/div[6]/div/div[4]/div/div[1]/div/div[1]/div/div[2]/a")
이미지버튼.click()

while True:
    current_height = driver.execute_script("return document.documentElement.scrollHeight")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    sleep(1)
    if driver.execute_script("return document.documentElement.scrollHeight") == current_height:
        break
    결과더보기버튼 = driver.find_element(By.CSS_SELECTOR, "input[value='결과 더보기']")
    try:
        결과더보기버튼.click()
    except ElementNotInteractableException as e:
        pass

이미지리스트 = driver.find_elements(By.CSS_SELECTOR, "span[jsslot] > div > div > div > a > div > img[width]")
이미지리스트 += driver.find_elements(By.CSS_SELECTOR, "span[jsslot] > div > div > div > div > a > div > img[width]")

for idx, img_tag in enumerate(이미지리스트):
    sleep(0.1)
    img_url = img_tag.get_attribute("src")
    try:
        with urllib.request.urlopen(img_url) as response:
            img = response.read()
            with open(os.path.join(검색어, f"{검색어}_{idx}.jpg"), mode="wb") as f:
                f.write(img)
        print(f"{검색어}_{idx}.jpg")

    except AttributeError as e:
        sleep(1)
        pass

driver.quit()