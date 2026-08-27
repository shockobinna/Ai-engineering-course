# import pyttsx3
# import speech_recognition as sr


# def speak(text):
#     print(f"JARVIS: {text}")

#     engine = pyttsx3.init('sapi5')

#     voices = engine.getProperty('voices')
#     engine.setProperty('voice', voices[1].id)

#     engine.say(text)
#     engine.runAndWait()
#     engine.stop()


# def take_command():
#     """
#     Take microphone input from the user
#     and return it as a string.
#     """

#     recognizer = sr.Recognizer()

#     with sr.Microphone() as source:
#         print("Listening...")

#         recognizer.pause_threshold = 1.5
#         audio = recognizer.listen(source)

#     try:
#         print("Recognizing...")

#         query = recognizer.recognize_google(
#             audio,
#             language="en-us"
#         )

#         print(f"User said: {query}\n")

#         return query.lower()

#     except Exception:
#         speak("Sorry, I didn't understand that.")
#         print("Sorry, I didn't understand that.")

#         return None

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


def take_command(silent=False, timeout=None):
    """
    Take microphone input from the user
    and return it as a string.

    silent=True:
        Don't speak when speech isn't understood.

    timeout:
        Maximum time to wait for the user to start speaking.
    """

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        print("Listening...")

        recognizer.pause_threshold = 1.5

        try:

            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=5
            )

        except sr.WaitTimeoutError:

            print("Listening timed out.")

            return None

    try:

        print("Recognizing...")

        query = recognizer.recognize_google(
            audio,
            language="en-us"
        )

        print(f"User said: {query}\n")

        return query.lower()

    except sr.UnknownValueError:

        if not silent:
            speak("Sorry, I didn't understand that.")

        print("Nothing understood.")

        return None

    except sr.RequestError as error:

        print(f"Speech recognition error: {error}")

        if not silent:
            speak("I'm having trouble connecting to the speech recognition service.")

        return None