from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import time

service = Service(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

driver.get("https://www.ncei.noaa.gov/data/dmsp-space-weather-sensors/access/")
#driver.get("https://youtube.com")
print(f'opened URL: {driver.current_url}')
folders = ['anch_9', 'anch_10', 'anch_11', 'anch_12', 'anch_13', 'anch_14', 
           'anch_15', 'anch_16', 'anch_17', 'anch_18', 'anch_19']
for idx in folders:
    time.sleep(3)
    driver.find_element('id', idx).click()
    time.sleep(3)
    driver.find_element('id', 'anch_6').click()
    time.sleep(3)
    driver.back()
    driver.back()

