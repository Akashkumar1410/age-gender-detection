import nltk
import random
from nltk.chat.util import Chat, reflections

# Define some reflections (for pronouns)
reflections = {
    "i am": "you are",
    "i was": "you were",
    "i": "you",
    "i'm": "you are",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "you are": "I am",
    "you were": "I was",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you",
}

# Define the chatbot's responses
pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, how can I help you today?",]
    ],
    [
        r"what is your name?",
        ["I am a chatbot.",]
    ],
    [
        r"how are you ?",
        ["I'm doing well, thank you! How can I assist you?",]
    ],
    [
        r"(.*) (help|support)",
        ["I can help you with various tasks. Just ask me a question.",]
    ],
    [
        r"quit",
        ["Goodbye!", "It was nice talking to you. Goodbye!"]
    ],
    [
        r"(.*)",
        ["I'm sorry, I don't understand. Please ask a different question.",]
    ],
]

# Create a chatbot instance
chatbot = Chat(pairs, reflections)

# Function to start and interact with the chatbot
def chat_with_bot():
    print("Hello! I'm a chatbot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = chatbot.respond(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    chat_with_bot()
