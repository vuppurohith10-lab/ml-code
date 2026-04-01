import spidev
import socket
import time
import RPi.GPIO as GPIO
from threading import Thread
from collections import deque

from luma.core.interface.serial import spi as oled_spi
from luma.oled.device import ssd1306
from PIL import Image, ImageDraw, ImageFont

# ---------------- GPIO SETUP ----------------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

ENA = 18
IN1 = 20
IN2 = 21

GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

pwm = GPIO.PWM(ENA, 1000)
pwm.start(0)

GPIO.output(IN1, GPIO.HIGH)
GPIO.output(IN2, GPIO.LOW)

# ---------------- MCP3008 (ADC) ----------------
spi_adc = spidev.SpiDev()
spi_adc.open(0, 0)
spi_adc.max_speed_hz = 1350000

def read_adc(channel=0):
    adc = spi_adc.xfer2([1, (8 + channel) << 4, 0])
    return ((adc[1] & 3) << 8) + adc[2]

# ---------------- OLED ----------------
serial = oled_spi(port=0, device=1, gpio_DC=25, gpio_RST=27)
device = ssd1306(serial)
font = ImageFont.load_default()

def show_message(msg):
    image = Image.new("1", (device.width, device.height))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, device.width, device.height), fill=0)
    draw.text((10, 25), msg, font=font, fill=255)
    device.display(image)

# ---------------- POT SMOOTHING ----------------
buffer = deque(maxlen=10)
current_speed = 0

def get_smooth_speed():
    raw = read_adc(0)
    buffer.append(raw)
    avg = sum(buffer) / len(buffer)
    return int((avg / 1023) * 100)

# ---------------- MOTOR ----------------
def set_speed(speed):
    pwm.ChangeDutyCycle(speed)

def gradual_speed(target):
    global current_speed
    while current_speed != target:
        if current_speed < target:
            current_speed += 1
        else:
            current_speed -= 1
        set_speed(current_speed)
        time.sleep(0.05)

# ---------------- SOCKET ----------------
HOST = '0.0.0.0'
PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print("🚀 Waiting for ML connection...")
conn, addr = server.accept()
print("Connected:", addr)

# ---------------- ML HANDLER ----------------
override_active = False

def handle_ml(label):
    global override_active

    override_active = True

    if label == "workinprogress":
        show_message("Work Zone")
        time.sleep(3)
        gradual_speed(20)

    elif label == "speedlimit":
        show_message("Speed 30")
        time.sleep(5)
        gradual_speed(30)

    elif label == "crossing":
        show_message("Pedestrian")
        time.sleep(3)

    elif label == "slowdown":
        show_message("Slow Down")
        time.sleep(3)

    override_active = False

# ---------------- MAIN LOOP ----------------
try:
    while True:

        # ---- Receive ML ----
        conn.settimeout(0.01)
        try:
            data = conn.recv(1024)
            if data:
                label = data.decode().strip()
                print("📩 ML:", label)
                Thread(target=handle_ml, args=(label,)).start()
        except:
            pass

        # ---- Pot Control ----
        if not override_active:
            target_speed = get_smooth_speed()

            if current_speed < target_speed:
                current_speed += 1
            elif current_speed > target_speed:
                current_speed -= 1

            set_speed(current_speed)
            show_message(f"Speed {current_speed}%")

        time.sleep(0.02)

except KeyboardInterrupt:
    print("Stopping...")

# ---------------- CLEANUP ----------------
pwm.stop()
GPIO.cleanup()
spi_adc.close()
conn.close()