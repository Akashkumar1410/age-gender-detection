import speech_recognition as sr
import keyboard
import time

listener = sr.Recognizer()
listening = False

def start_listening():
    global listening
    listening = True
    print('Listening...')

def stop_listening():
    global listening
    if listening:
        listening = False
        print('Stopped listening')

while True:
    if keyboard.is_pressed('L') and not listening:
        start_listening()
        with sr.Microphone() as source:
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            print('You said:', command)
    if keyboard.is_pressed('S'):
        stop_listening()
        # Add a delay to prevent multiple 'S' presses in quick succession
        time.sleep(0.2)
