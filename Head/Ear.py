import speech_recognition as sr
import os
import threading
from mtranslate import translate
from colorama import Fore, Style, init
from speech_recognition import recognizers

init(autoreset=True)
def print_loop():
    while True:
        print(Fore.LIGHTGREEN_EX + "I am Litening...", end="", flush=True)
        print(Style.RESET_ALL, end="", flush=True)
        print("", end="", flush=True)

def Trans_hindi_to_English(text):
    english_text = translate(text, to_language="en-in")
    return english_text

def listen():
    r = sr.Recognizer()
    r.dynamic_energy_threshold = False
    r.energy_threshold = 35000
    r.dynamic_energy_adjustment_damping = 0.03
    r.dynamic_energy_ratio = 1.9
    r.pause_threshold = 0.4
    r.operation_timeout = None
    r.pause_threshold = 0.2
    r.non_speaking_duration = 0.3

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        while True:
            print(Fore.LIGHTGREEN_EX + "I am Litening...", end="", flush=True)
            try:
                audio = r.listen(source, timeout=None)
                print("\r"+ Fore.LIGHTYELLOW_EX + "Got it , Recognizing...", end="", flush=True)
                recognized_txt = r.recognize_amazon(audio).lower()
                if recognized_txt :
                    translated_txt = Trans_hindi_to_English(recognized_txt)
                    print("\r"+Fore.Blue + "Mr Kalgi : " + translated_txt)
                    return translated_txt
                else:
                    return ""
            except sr.UnknownValueError:
                recognized_txt = ""
            finally:
                print("\r", end="", flush=True)

            os.system("cls" if os.name == "nt" else "clear")
            listen_thread = threading.Thread(target=listen)
            print_thread = threading.Thread(target=print_loop)
            listen_thread.start()
            print_thread.start()
            listen_thread.join()
            print_thread.join()


listen()