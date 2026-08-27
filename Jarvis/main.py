# from voice import speak, take_command
# from commands import process_command
# from datetime import datetime


# author = "David"


# def wish_me():

#     hour = datetime.now().hour

#     if 0 <= hour < 12:
#         speak(f"Good Morning {author}")

#     elif 12 <= hour < 18:
#         speak(f"Good Afternoon {author}")

#     else:
#         speak(f"Good Evening {author}")

#     speak(f"Hello {author}, I am Jarvis.How may I help ypu?")


# def main():

#     wish_me()

#     while True:

#         query = take_command()

#         if query is None:
#             continue

#         if not process_command(query):
#             break


# if __name__ == "__main__":
#     main()

from voice import speak, take_command
from commands import process_command
from datetime import datetime


author = "David"

# How long Jarvis stays awake after the last command
CONVERSATION_TIMEOUT = 30


def wish_me():

    hour = datetime.now().hour

    if 0 <= hour < 12:
        speak(f"Good Morning {author}")

    elif 12 <= hour < 18:
        speak(f"Good Afternoon {author}")

    else:
        speak(f"Good Evening {author}")


def main():

    wish_me()

    speak(f"Hello {author}, I am Jarvis.")

    while True:

        # =====================================
        # SLEEPING
        # =====================================

        print("\n--------------------------------")
        print("JARVIS IS SLEEPING")
        print("Say 'Jarvis' to wake me.")
        print("--------------------------------")

        query = take_command(
            silent=True
        )

        if query is None:
            continue

        # Did we hear "Jarvis"?
        if "jarvis" not in query:
            continue

        # =====================================
        # JARVIS WAKES UP
        # =====================================

        speak("Yes, how may I help you?")

        # =====================================
        # CONVERSATION MODE
        # =====================================

        while True:

            print("\n--------------------------------")
            print("JARVIS IS AWAKE")
            print(f"Waiting for command ({CONVERSATION_TIMEOUT}s)...")
            print("--------------------------------")

            command = take_command(
                silent=True,
                timeout=CONVERSATION_TIMEOUT
            )

            # ---------------------------------
            # TIMEOUT = GO BACK TO SLEEP
            # ---------------------------------

            if command is None:
                print("No command detected.")
                print("Jarvis is going back to sleep.")
                break

            # ---------------------------------
            # PROCESS COMMAND
            # ---------------------------------

            result = process_command(command)

            print(f"process_command returned: {result}")

            # ---------------------------------
            # EXIT PROGRAM
            # ---------------------------------

            if result is False:
                print("Jarvis is shutting down.")
                return


if __name__ == "__main__":
    main()