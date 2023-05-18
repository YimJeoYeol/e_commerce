import os
from time import sleep
import urllib

from selenium import webdriver
import chromedriver_autoinstaller
from selenium.common import ElementNotInteractableException
from selenium.webdriver.common.by import By

chromedriver_autoinstaller.install()

검색어리스트 = ["삼겹살", "치킨", "된장찌개", "김치", "돌솥비빔밥", "제육볶음"]
for 검색어 in 검색어리스트:
    try:
        os.mkdir(검색어)
    except FileExistsError:
        pass
    driver = webdriver.Chrome()

    driver.get("https://www.google.com")

    # 검색창을 찾아줘. 태그명은 textarea, title 속성값은 "검색"인 요소야.
    검색창 = driver.find_element(By.CSS_SELECTOR, "textarea[title='검색']")

    # 검색창에 검색어를 넣고 실행해
    검색창.send_keys(검색어)
    검색창.submit()

    이미지버튼 = driver.find_element(By.XPATH, "/html/body/div[6]/div/div[4]/div/div[1]/div/div[1]/div/div[2]/a")
    이미지버튼.click()

    # 스크롤을 계속 내리다가, "결과 더보기" 버튼이 보이면 클릭, 콘텐츠 바닥까지 가면 while문 종료
    while True:
        현재_스크롤_길이 = driver.execute_script("return document.documentElement.scrollHeight")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(3)

        try:
            결과더보기버튼 = driver.find_element(By.CSS_SELECTOR, "input[value='결과 더보기']")
            결과더보기버튼.click()
        except ElementNotInteractableException:
            pass
        # 스크롤을 내려도 더 안내려가면 while문 종료
        sleep(3)
        if 현재_스크롤_길이 == driver.execute_script("return document.documentElement.scrollHeight"):
            break

    이미지리스트 = driver.find_elements(By.CSS_SELECTOR, "div > div > div > a > div > img[width]")
    이미지리스트 += driver.find_elements(By.CSS_SELECTOR, "div > div > div > div > a > div > img[width]")
    소스리스트 = [i.get_attribute("src") for i in 이미지리스트]

    전체길이 = len(이미지리스트)

    for idx, img in enumerate(소스리스트):
        try:
            with urllib.request.urlopen(img) as response:
                image = response.read()
                with open(os.path.join(검색어, f"{검색어}_{idx}.jpg"), "wb") as f:
                    f.write(image)
        except (AttributeError, urllib.error.URLError):
            pass
        print(f"[{idx+1}/{전체길이}] {검색어}_{idx}.jpg")
    driver.quit()