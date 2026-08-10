import wikipedia
from voice import speak


def search_wikipedia(query):

    speak("Searching Wikipedia...")

    query = query.replace("wikipedia", "")
    query = query.replace("who", "")
    query = query.replace("is", "")
    query = query.strip()

    print(f"Searching Wikipedia for: {query}")

    try:

        results = wikipedia.summary(
            query,
            sentences=2
        )

        speak("According to Wikipedia")

        print(results)

        speak(results)

    except wikipedia.exceptions.DisambiguationError:
        speak(
            "I found multiple results. "
            "Could you be more specific?"
        )

    except wikipedia.exceptions.PageError:
        speak(
            "I couldn't find that on Wikipedia."
        )