import json
import pyttsx3
import datetime
import speech_recognition as sr
import wikipedia
import requests
import json
import webbrowser
import os
import pywhatkit as kit
import smtplib
engine = pyttsx3.init('sapi5')

voices = engine.getProperty('voices')

# print(voices)
engine.setProperty('voice', voices[1].id)
# print(voices[0].id)

author = "David"


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


# def sendEmail(to, content):
#     server = smtplib.SMTP('smtp.gmail.com', 587)
#     server.ehlo()
#     server.starttls()
#     server.login('Your Email', 'Password')
#     server.sendmail('Your Email', to, content)
#     server.close()


def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak(f"Good Morning {author}")
    elif hour >= 12 and hour < 18:
        speak(f"Good Afternoon {author}")
    else:
        speak(f"Good Evening {author}")

    speak(f"Hello {author} I am Jarvis, Please tell me How may I help You")


# def takeCommend():
#     '''
#     take microphone input from the user and return string
#     '''
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Listening...")
#         r.pause_threshold = 1.5
#         audio = r.listen(source)
#     try:
#         print("Recognizing...")
#         query = r.recognize_google(audio, language='en-in')
#         print(f"User Said:{query} \n")
#     except Exception as e:
#         print(f"Sorry {author}, Say That again... ")
#         return "None"
#     return query


if __name__ == "__main__":
    # speak(f"Welcome {author}, I am a Jarvis Female version")
    wishMe()
#     # takeCommend()
#     if 1:
#         query = takeCommend().lower()
#         if 'wikipedia' and 'who' in query:
#             speak("Searching Wikipedia...")
#             query = query.replace("wikipedia", "")
#             results = wikipedia.summary(query, sentences=2)
#             speak("According to wikipedia")
#             print(results)
#             speak(results)

#         elif 'news' in query:
#             speak("News Headlines")
#             query = query.replace("news", "")
#             url = "Your News Api url"
#             news = requests.get(url).text
#             news = json.loads(news)
#             art = news['articles']
#             for article in art:
#                 print(article['title'])
#                 speak(article['title'])

#                 print(article['description'])
#                 speak(article['description'])
#                 speak("Moving on to the next news")
#         elif 'open google' in query:
#             webbrowser.open("google.com")

#         elif 'open youtube' in query:
#             webbrowser.open("youtube.com")

#         elif 'search browser' in query:
#             speak("What should i search ?")
#             um = takeCommend().lower()
#             webbrowser.open(f"{um}")

#         elif 'ip address' in query:
#             ip = requests.get('http://api.ipify.org').text
#             print(f"Your ip is {ip}")
#             speak(f"Your ip is {ip}")

#         elif 'open command prompt' in query:
#             os.system("start cmd")

#         elif 'open photoshop' in query:
#             codepath = "C:\\Program Files\\Adobe\\Adobe Photoshop 2021\\Photoshop.exe"
#             os.startfile(codepath)

#         elif 'open code' in query:
#             codepath = "C:\\Users\\MyPc\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"
#             os.startfile(codepath)

#         elif 'play music' in query:
#             music_dir = 'D:\\jarvis\\music'
#             songs = os.listdir(music_dir)
#             print(songs)
#             os.startfile(os.path.join(music_dir, songs[0]))
#         elif 'play youtube' in query:
#             speak("What should i search in youtube ?")
#             cm = takeCommend().lower()
#             kit.playonyt(f"{cm}")
#         elif 'send message' in query:
#             speak("Who do you want to send the message ?")
#             num = input("Enter number : \n")
#             speak("what do you want to send?")
#             msg = takeCommend().lower()
#             speak("Please Enter Time sir.")
#             H = int(input("Enter hour ?\n"))
#             M = int(input("Enter Minutes ?\n"))
#             kit.sendwhatmsg(num, msg, H, M)
#         elif 'send email' in query:
#             speak("What should i send sir ?")
#             content = takeCommend().lower()
#             speak("Whom to send the email , enter email address sir ")
#             to = input("Enter Email Address : \n ")
#             sendEmail(to, content)
