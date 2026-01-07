from HueWrapper import HueController
import time

# Your bridge info
BRIDGE_IP = "192.168.1.48"
USERNAME = "958lVW8ABIsBp5r6fOpabdgkiZq3Kh-4PoerQ1bD"

hue = HueController(BRIDGE_IP, USERNAME)

hue.list_lights()
#hue.set_brightness("plafond", 50)
#hue.set_color("plafond", 42005, 200)

target_lamp = "Hue color lamp"

hue.turn_on(target_lamp)
hue.set_brightness(target_lamp, 100)
hue.set_color_temperature(target_lamp, 490)
#hue.animate_breath(target_lamp, speed= 0.008)
#hue.turn_off(target_lamp)
#hue.cycle_colors(target_lamp)

"""
lamp = "Desk Lamp"

hue.turn_on(lamp)
hue.set_brightness(lamp, 180)
hue.set_color(lamp, 10000, 200)
hue.turn_off(lamp)
time.sleep(2)
"""

