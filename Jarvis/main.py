# import json
# import pyttsx3
# import datetime
# import speech_recognition as sr
# import wikipedia
# import requests
# import json
# import webbrowser
# import os
# import pywhatkit as kit
# import smtplib
# import time

# # engine = pyttsx3.init('sapi5')

# # voices = engine.getProperty('voices')

# # # print(voices)
# # engine.setProperty('voice', voices[1].id)
# # # print(voices[0].id)

# author = "David"

# def speak(text):
#     print(f"JARVIS: {text}")

#     engine = pyttsx3.init('sapi5')

#     voices = engine.getProperty('voices')
#     engine.setProperty('voice', voices[1].id)

#     engine.say(text)
#     engine.runAndWait()
#     engine.stop()


# def wishMe():

#     hour = datetime.datetime.now().hour

#     print("[1] Before greeting")

#     if 0 <= hour < 12:
#         speak(f"Good Morning {author}")

#     elif 12 <= hour < 18:
#         speak(f"Good Afternoon {author}")

#     else:
#         speak(f"Good Evening {author}")

#     print("[2] Greeting finished")

#     speak(f"Hello {author} I am Jarvis, Please tell me How may I help You")

#     print("[3] Finished")






# # def sendEmail(to, content):
# #     server = smtplib.SMTP('smtp.gmail.com', 587)
# #     server.ehlo()
# #     server.starttls()
# #     server.login('Your Email', 'Password')
# #     server.sendmail('Your Email', to, content)
# #     server.close()



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
#         query = r.recognize_google(audio, language='en-us')
#         print(f"User Said:{query} \n")
#     except Exception as e:
#         print(f"Sorry {author}, Say That again... ")
#         return "None"
#     return query


# if __name__ == "__main__":
#     # speak(f"Welcome {author}, I am a Jarvis Female version")
#     wishMe()
#     # takeCommend()
#     if 1:
#         query = takeCommend().lower()
#         if 'wikipedia' in query or 'who' in query:
#             speak("Searching Wikipedia...")
#             query = query.replace("wikipedia", "")
#             results = wikipedia.summary(query, sentences=2)
#             speak("According to wikipedia")
#             print(results)
#             speak(results)

#         elif 'news' in query:
#             speak("News Headlines")
#             query = query.replace("news", "")
#             url = "https://newsapi.org/v2/top-headlines?country=ng&apiKey=58770e505aec47a9abb06198affb255c"
#             news = requests.get(url).text
#             news = json.loads(news)
#             art = news['articles']
#             print(f"Here is the news: {art}")
#             if art:
#                 for article in art:
#                     print(article['title'])
#                     speak(article['title'])

#                     print(article['description'])
#                     speak(article['description'])
#                     speak("Moving on to the next news")
#             else:
#                 speak("No news found")
                
# #         elif 'open google' in query:
# #             webbrowser.open("google.com")

# #         elif 'open youtube' in query:
# #             webbrowser.open("youtube.com")

# #         elif 'search browser' in query:
# #             speak("What should i search ?")
# #             um = takeCommend().lower()
# #             webbrowser.open(f"{um}")

# #         elif 'ip address' in query:
# #             ip = requests.get('http://api.ipify.org').text
# #             print(f"Your ip is {ip}")
# #             speak(f"Your ip is {ip}")

# #         elif 'open command prompt' in query:
# #             os.system("start cmd")

# #         elif 'open photoshop' in query:
# #             codepath = "C:\\Program Files\\Adobe\\Adobe Photoshop 2021\\Photoshop.exe"
# #             os.startfile(codepath)

# #         elif 'open code' in query:
# #             codepath = "C:\\Users\\MyPc\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"
# #             os.startfile(codepath)

# #         elif 'play music' in query:
# #             music_dir = 'D:\\jarvis\\music'
# #             songs = os.listdir(music_dir)
# #             print(songs)
# #             os.startfile(os.path.join(music_dir, songs[0]))
# #         elif 'play youtube' in query:
# #             speak("What should i search in youtube ?")
# #             cm = takeCommend().lower()
# #             kit.playonyt(f"{cm}")
# #         elif 'send message' in query:
# #             speak("Who do you want to send the message ?")
# #             num = input("Enter number : \n")
# #             speak("what do you want to send?")
# #             msg = takeCommend().lower()
# #             speak("Please Enter Time sir.")
# #             H = int(input("Enter hour ?\n"))
# #             M = int(input("Enter Minutes ?\n"))
# #             kit.sendwhatmsg(num, msg, H, M)
# #         elif 'send email' in query:
# #             speak("What should i send sir ?")
# #             content = takeCommend().lower()
# #             speak("Whom to send the email , enter email address sir ")
# #             to = input("Enter Email Address : \n ")
# #             sendEmail(to, content)

from voice import speak, take_command
from commands import process_command
from datetime import datetime


author = "David"


def wish_me():

    hour = datetime.now().hour

    if 0 <= hour < 12:
        speak(f"Good Morning {author}")

    elif 12 <= hour < 18:
        speak(f"Good Afternoon {author}")

    else:
        speak(f"Good Evening {author}")

    speak(f"Hello {author}, I am Jarvis.How may I help ypu?")


def main():

    wish_me()

    while True:

        query = take_command()

        if query is None:
            continue

        if not process_command(query):
            break


if __name__ == "__main__":
    main()