import pyautogui as pg
import random
import pyperclip  # Required to copy and paste emojis

# Simulate receiving a message (replace this with actual received message)
received_message = "Good morning😁"  # Replace with actual received message

# Select a random emoji from the list
emojis = ["😀", "😁"]
random_emoji = random.choice(emojis)

if "Good morning" in received_message.lower():
    # Copy the emoji to the clipboard
    pyperclip.copy(random_emoji)

    # Paste the emoji into the message input box
    pg.hotkey("ctrl", "v")

    # Write the response message and press Enter
    response_message = f"Hello, how are you? {random_emoji}"
    pg.write(response_message)
    pg.hotkey("ctrl", "v")
    pg.press("Enter")
else:
    # Select a random message from the list
    message_options = [
        f"Kon hai tu {random_emoji}",
        f"jldi bta kon hai tu {random_emoji}",
        f"call me {random_emoji}"
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