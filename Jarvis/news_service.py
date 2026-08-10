import os
import requests
from voice import speak


NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def get_news():

    speak("Here are the latest news headlines.")

    url = (
        "https://newsapi.org/v2/top-headlines"
        f"?country=ng&apiKey={NEWS_API_KEY}"
    )

    try:

        response = requests.get(url)

        news = response.json()

        articles = news.get("articles", [])

        if not articles:
            speak("I couldn't find any news.")
            return

        for article in articles:

            title = article.get("title")

            if title:
                print(title)
                speak(title)

        speak("Those are the latest headlines.")

    except Exception as e:

        print(f"News error: {e}")

        speak(
            "Sorry, I couldn't retrieve the news."
        )