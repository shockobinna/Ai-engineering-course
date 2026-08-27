from voice import speak, take_command
from wikipedia_service import search_wikipedia
from news_service import get_news
from local_aplication import local_IP, open_cmd, open_vault,open_anotation,play_songs_locally
from browser_service import (
    open_gmail,
    open_youtube,
    open_google,
    search_browser,
    batida_online,
    cervello,
    play_song_on_youtube,
    send_whats_msg,
    send_email
)


def process_command(query):

    if "wikipedia" in query or "who is" in query:

        search_wikipedia(query)

    elif "news" in query:

        get_news()

    elif "open gmail" in query:

        open_gmail()

    elif "open youtube" in query:

        open_youtube()

    elif "open google" in query:

        open_google()

    elif "search browser" in query:
        speak("What should i serach?")
        search_word = take_command()
        search_browser(search_word)

    elif "ip address" in query:
        local_IP()

    elif "open command prompt" in query or "open cmd" in query:
        speak("Opening command prompt.")
        open_cmd()

    elif "open keepass" in query or "abrir keepass" in query:
        speak("Opening password vault")
        open_vault()

    elif "open batida" in query:
        speak("Opening Batida online")
        batida_online()

    elif "open cases" in query:
        speak("Opening cervello in your browser")
        cervello()

    elif "open my annotation" in query or "annotation" in query or "notes" in query:
        speak("Opening you notes in file explorer")
        open_anotation()

    elif "play music" in query:
        speak("Enjoy the music")
        play_songs_locally()

    elif "play youtube" in query:
        play_song_on_youtube()

    elif "send message" in query:
        send_whats_msg()

    elif "send email" in query:
        send_email()


    elif (
    "goodbye" in query
    or "exit" in query
    or "thank you" in query
    or "thanks" in query
    or "quit" in query
):

        print("EXIT COMMAND DETECTED")

        speak("Goodbye, David.")

        print("Returning False to main.py")

        return False

    else:

        speak(
            "I'm sorry, I don't know how to do that yet. "
            "To exit or quit, please say goodbye or exit."
        )

    return True

#TO DO
# COMMANDS = {
#     "wikipedia": search_wikipedia,
#     "news": get_news,
#     "open gmail": open_gmail,
#     "open youtube": open_youtube,
#     "open cmd": open_cmd,
# }

# 🧠 JARVIS
# ├── What can you do?
# ├── Remember...
# ├── What do you remember?
# └── Forget...

# ⏰ TIME
# ├── Time
# ├── Date
# ├── Day
# ├── Alarm
# └── Reminder

# 💻 SYSTEM
# ├── CPU
# ├── RAM
# ├── Disk
# ├── Battery
# ├── Screenshot
# ├── Volume
# ├── Lock
# └── Shutdown

# 📁 FILES
# ├── Find file
# ├── Open folder
# ├── Create folder
# ├── Create file
# └── Read clipboard

# 🌦️ INFORMATION
# ├── Weather
# ├── News
# ├── Wikipedia
# └── Google

# 📧 EMAIL
# ├── Open Gmail
# ├── Check email
# ├── Read email
# ├── Send email
# └── Search email

# My suggested next build order

# Phase 1 — Core JARVIS

# Time/date
# Application launcher
# System information
# Screenshot
# File management

# Phase 2 — Intelligence
# 6. Weather
# 7. Gmail API
# 8. Calendar
# 9. Reminders
# 10. Memory

# Phase 3 — JARVIS experience
# 11. Wake word
# 12. Conversation mode
# 13. Context-aware commands
# 14. Confirmation for dangerous actions
# 15. Better natural-language command interpretation

# And Phase 4 is where things get really interesting: replace the giant if/elif router with an intent/command system powered by an LLM, so instead of requiring exact phrases like "open gmail", JARVIS understands:

# "Could you pull up my inbox?"

# and maps that to open_gmail() automatically.

# That's the point where your project goes from a Python voice-command tutorial to an actual AI desktop assistant. 🤖