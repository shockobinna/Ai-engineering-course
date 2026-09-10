import os
import subprocess
import requests
from voice import speak, take_command
from datetime import datetime
import psutil
import pyautogui
import subprocess
import os
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

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

def get_cpu():
    cpu = psutil.cpu_percent(interval=1)

    speak(f"CPU usage is {cpu} percent.")

def get_ram():
    ram = psutil.virtual_memory()

    used = ram.used / (1024 ** 3)
    total = ram.total / (1024 ** 3)
    percent = ram.percent

    speak(
        f"RAM usage is {percent} percent. "
        f"You are using {used:.1f} gigabytes out of {total:.1f}."
    )

def get_disk():
    disk = psutil.disk_usage("C:\\")

    used = disk.used / (1024 ** 3)
    total = disk.total / (1024 ** 3)
    percent = disk.percent

    speak(
        f"Disk usage is {percent} percent. "
        f"You are using {used:.1f} gigabytes out of {total:.1f}."
    )

def get_battery():
    battery = psutil.sensors_battery()

    if battery is None:
        speak("I couldn't detect a battery.")
        return

    percent = battery.percent

    if battery.power_plugged:
        speak(f"Battery is at {percent} percent and the computer is charging.")
    else:
        speak(f"Battery is at {percent} percent.")



def take_screenshot():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"screenshot_{timestamp}.png"

    pyautogui.screenshot(filename)

    speak("Screenshot taken.")

def lock_pc():
    speak("Locking the computer.")

    subprocess.run(
        ["rundll32.exe", "user32.dll,LockWorkStation"]
    )

def shutdown_pc():
    speak("Are you sure you want to shut down the computer?")

    # Replace this with however your JARVIS listens for responses
    confirmation = take_command()

    if "yes" in confirmation:
        speak("Shutting down the computer.")
        subprocess.run(["shutdown", "/s", "/t", "5"])
    else:
        speak("Shutdown cancelled.")



def get_volume():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_,
        CLSCTX_ALL,
        None
    )

    volume = interface.QueryInterface(
        IAudioEndpointVolume
    )

    current = volume.GetMasterVolumeLevelScalar()
    percent = int(current * 100)

    speak(f"Volume is at {percent} percent.")

def set_volume(percent):
    percent = max(0, min(100, percent))

    devices = AudioUtilities.GetSpeakers()

    interface = devices.Activate(
        IAudioEndpointVolume._iid_,
        CLSCTX_ALL,
        None
    )

    volume = interface.QueryInterface(
        IAudioEndpointVolume
    )

    volume.SetMasterVolumeLevelScalar(
        percent / 100,
        None
    )

    speak(f"Volume set to {percent} percent.")