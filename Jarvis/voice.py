import pyttsx3
import speech_recognition as sr


def speak(text):
    print(f"JARVIS: {text}")

    engine = pyttsx3.init('sapi5')

    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)

    engine.say(text)
    engine.runAndWait()
    engine.stop()


def take_command():
    """
    Take microphone input from the user
    and return it as a string.
    """

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")

        recognizer.pause_threshold = 1.5
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")

        query = recognizer.recognize_google(
            audio,
            language="en-us"
        )

        print(f"User said: {query}\n")

        return query.lower()

    except Exception:
        speak("Sorry, I didn't understand that.")
        print("Sorry, I didn't understand that.")

        return None