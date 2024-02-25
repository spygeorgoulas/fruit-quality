import RPi.GPIO as GPIO
import time
from picamera import PiCamera
from time import sleep
import requests

camera = PiCamera()
url = 'http ://94.65.90.245:6060/ upload'

BUTTON_PIN = 16
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN , GPIO.IN, pull_up_down=GPIO.PUD_UP)

try:
while True:
time.sleep(0.1)
if GPIO.input(BUTTON_PIN) == GPIO.LOW:
 camera.start_preview(alpha=192)
 camera.capture("/home/pi/Desktop/buttonimage.jpg")
 camera.stop_preview()
 files = {'image': open('/home/pi/Desktop/buttonimage.jpg', 'rb'
)}
 response = requests.post(url, files=files)
 print(response.text)
 except KeyboardInterrupt:
 GPIO.cleanup()
