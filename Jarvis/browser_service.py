import os
from dotenv import load_dotenv
import webbrowser
from urllib.parse import quote
from voice import speak, take_command
import pywhatkit as kit
import smtplib

load_dotenv()


def open_gmail():

    speak("Opening Gmail.")

    webbrowser.open(
        "https://mail.google.com"
    )


def open_youtube():

    speak("Opening YouTube.")

    webbrowser.open(
        "https://www.youtube.com"
    )


def open_google():

    speak("Opening Google.")

    webbrowser.open(
        "https://www.google.com"
    )

def search_browser(query):

    speak(f"Searching for {query} as requested")
    search_query = quote(query)

    webbrowser.open(
        f"https://www.google.com/search?q={search_query}"
    )

def batida_online():

    webbrowser.open(
        f"https://hora-certa.bettaglobal.com.br/meus-lancamentos"
    )

def cervello():
    webbrowser.open(
        f"https://www.cervelloesm.com.br/Betta/Atendimento/Home"
    )

def play_song_on_youtube():
    speak("What should i search in youtube ?")
    cm = take_command()
    kit.playonyt(f"{cm}")

def send_whats_msg():
    speak("Who do you want to send the message ?")
    num = input("Enter number : \n")
    speak("what do you want to send?")
    msg = take_command()
    speak("Please Enter Time sir.")
    H = int(input("Enter hour ?\n"))
    M = int(input("Enter Minutes ?\n"))
    kit.sendwhatmsg(num, msg, H, M)

def sendEmail(to, content):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login('Your Email', 'Password')
    server.sendmail('Your Email', to, content)
    server.close()

def send_email():
    speak("What should I send sir ?")
    content = take_command()
    speak("Whom to send the email , enter email address sir ")
    to = input("Enter Email Address : \n ")
    sendEmail(to, content)

import requests

def get_weather():
    api_key = os.getenv("WEATHER_API_KEY")
    city = "São Paulo"

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=en"

    response = requests.get(url)
    data = response.json()

    temperature = data["main"]["temp"]
    description = data["weather"][0]["description"]

    speak(f"The temperature in {city} is {temperature:.0f} degrees Celsius, with {description}.")

    