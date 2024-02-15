from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

# Initialize the Chrome WebDriver
driver = webdriver.Chrome(r'C:\Users\HOME\PycharmProjects\chromedriver_win32\chromedriver.exe')  # Replace with the correct path to chromedriver.exe

# Open YouTube in the browser
driver.get("https://www.youtube.com")

# Find and play a video
video = driver.find_element_by_id("punjabi songs")  # Replace with the ID of the video you want to play
video.click()

# Control media playback
time.sleep(5)  # Allow time for the video to load

# Pause/Play the video (press the spacebar)
driver.find_element_by_tag_name('body').send_keys(Keys.SPACE)
time.sleep(2)  # Wait for 2 seconds

# Skip to the next video (press 'l' key)
driver.find_element_by_tag_name('body').send_keys('l')
time.sleep(2)  # Wait for 2 seconds

# Go back to the previous video (press 'k' key)
driver.find_element_by_tag_name('body').send_keys('k')
time.sleep(2)  # Wait for 2 seconds

# Seek forward (press 'right arrow' key)
driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_RIGHT)
time.sleep(2)  # Wait for 2 seconds

# Seek backward (press 'left arrow' key)
driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_LEFT)
time.sleep(2)  # Wait for 2 seconds

# Close the browser
driver.quit()
