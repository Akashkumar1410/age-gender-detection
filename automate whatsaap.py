import time
import pywhatkit
import random

# country code with mobile number iof thr person
recipient_number = 'Country-code+mobile number '

# List of emojis to choose from  this list
emojis = ["😊", "❤️", "😘", "🥰", "😍", "🌟"]

# Select a random emoji from the list  and attached with the messaage
random_emoji = random.choice(emojis)

# Message to send
message = [f" hii how are you {random_emoji}",
           f" what you doing  {random_emoji}",
           f"pick up the call {random_emoji}"
           ]
random_message = random.choice(message)

# Send the message at 17:08 replace the time  of hours and minutes
pywhatkit.sendwhatmsg(recipient_number, random_message,0,0)

