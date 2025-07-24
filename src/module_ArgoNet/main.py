
import time
import keyboard
import threading

print('\033[H\033[J', end='\n')
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

while True:
    loading = [' ',' ',' ',' ',' ']
    for i in range(0, 5):
        print('\033[H')
        print(GREEN + f'system restart\nmenu>>>\n')
        print(f'   audio devices    \n')
        print(f'>>>camera system    {"".join("     ")}\n')
        print(f'   ventilation      \n')
        print(f'   reboot all')
        print(f'   exit\n' + RESET)
        time.sleep(0.25)

        print('\033[H')
        print(GREEN + f'system restart\nmenu>>>\n')
        print(f'   audio devices    \n')
        print(f'>>>camera system    {"".join(RED + "error" + GREEN)}\n')
        loading[i-1] = ' '
        loading[i] = '█'
        print(f'   ventilation      {"".join(loading)}\n')
        print(f'   reboot all')
        print(f'   exit\n' + RESET)
        time.sleep(0.25)
