from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

url = "https://asx-options-calc-myfvbqtgkfpzuwurid6jce.streamlit.app/"

options = Options()
options.add_argument("--headless") # Runs the browser in the background
driver = webdriver.Chrome(options=options)

try:
    driver.get(url)
    # Check if the app is asleep and the wake button is present
    button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'Yes, get this app back up')]"))
    )
    button.click()
    print("App was asleep. Woke it up!")
except Exception:
    print("App is already awake or button not found. Traffic generated!")
finally:
    driver.quit()
