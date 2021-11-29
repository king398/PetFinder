import pyautogui
import time
from tqdm import tqdm
import random

list = ["https://tenor.com/view/explosion-boom-gif-8911363",
        "https://tenor.com/view/russia-soviet-missile-missile-truck-rocket-truck-gif-18785324",
        "https://tenor.com/view/america-aircraft-carrier-carrier-ocean-jets-gif-8168390",
        "https://tenor.com/view/mlrs-rockets-artillery-army-military-gif-11288811",
        "https://tenor.com/view/normandy-ww2-operation-overload-france-gif-5135131",
        "https://tenor.com/view/f22-lockheed-martin-f22raptor-plane-jet-plane-gif-15000947",
        "https://tenor.com/view/viva-la-revolution-gif-11760395", "https://i.makeagif.com/media/10-08-2017/b83FzR.mp4",
        "https://images-cdn.9gag.com/photo/aZ0O3D6_700b.jpg"]
for i in tqdm(range(200)):
	pyautogui.typewrite(f"horni army is attacking {random.choice(list)}")
	pyautogui.press("enter")
	time.sleep(2)
	print(i)
ps:/