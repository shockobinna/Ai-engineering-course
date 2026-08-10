import os
import subprocess
import requests
from voice import speak

def local_IP():
    speak(f"Checking your ip address")
    ip = requests.get('http://api.ipify.org').text
    print(f"Your ip is {ip}")
    speak(f"Your ip is {ip}")

def open_cmd():
    subprocess.Popen(
        ["cmd.exe"],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

def open_vault():
    codepath = "C:\\Program Files\\KeePassXC\\KeePassXC.exe"
    os.startfile(codepath)

def open_anotation():
    notepath = "C:\\Users\\lekwuwa.okorie\\Downloads\\my_anotation"
    os.startfile(notepath)

def play_songs_locally():
    music_dir = 'D:\\jarvis\\music'
    songs = os.listdir(music_dir)
    print(songs)
    os.startfile(os.path.join(music_dir, songs[0]))
