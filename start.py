import subprocess
import os
subprocess.call(["pip", "install", "-r", "assets/requirements.txt"])
from utils import inputs
os.system('cls' if os.name == 'nt' else 'clear')
print(r'''
  /$$$$$$  /$$$$$$ /$$      /$$ /$$
 /$$__  $$|_  $$_/| $$$    /$$$|__/
| $$  \ $$  | $$  | $$$$  /$$$$ /$$
| $$$$$$$$  | $$  | $$ $$/$$ $$| $$
| $$__  $$  | $$  | $$  $$$| $$| $$
| $$  | $$  | $$  | $$\  $ | $$| $$
| $$  | $$ /$$$$$$| $$ \/  | $$| $$
|__/  |__/|______/|__/     |__/|__/
''')
inputs.aimbot(ENABLE_AIMBOT=True)
