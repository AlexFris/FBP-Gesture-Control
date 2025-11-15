from PhillipsHueWrapperModule import HueController
import time

# Your bridge info
BRIDGE_IP = "192.168.178.213"
USERNAME = "NdzQVUnpZAsG21NTQOS932ilAKYTX2UFdWwNZ4gF"

hue = HueController(BRIDGE_IP, USERNAME)

hue.list_lights()
#hue.list_lights()
#hue.set_brightness("plafond", 50)
#hue.set_color("plafond", 42005, 200)

hue.turn_on("plafond")
hue.cycle_colors("plafond")

"""
lamp = "Desk Lamp"

hue.turn_on(lamp)
hue.set_brightness(lamp, 180)
hue.set_color(lamp, 10000, 200)
hue.turn_off(lamp)
time.sleep(2)
"""

