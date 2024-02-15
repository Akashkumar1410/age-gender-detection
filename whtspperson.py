import pyautogui as pg
import time
import random
import pyperclip  # Required to copy and paste emojis

time.sleep(2)

for _ in range(1000):
    # Select a random emoji from the list
    emojis = ["😀","😁",]

    random_emoji = random.choice(emojis)

    message_options = [
        f"joiya   kya haal hai  {random_emoji}",
        f"kya karr rha hai  {random_emoji}",
        f"tum chutioya ho kya {random_emoji}"
    ]

    # Select a random message from the list
    random_message = random.choice(message_options)

    # Copy the emoji to the clipboard
    pyperclip.copy(random_emoji)

    # Paste the emoji into the message input box
    pg.hotkey("ctrl", "v")

    # Write the rest of the message and press Enter
    pg.write(random_message)
    pg.hotkey("ctrl", "v")
    pg.press("Enter")

    time.sleep(0)  # Add a delay between messages
