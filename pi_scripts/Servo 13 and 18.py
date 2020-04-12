#!/usr/bin/env python3

import time
import RPi.GPIO as gp

servo_pin1 = 18
servo_pin2 = 13

gp.setmode(gp.BCM)

gp.setup(servo_pin1, gp.OUT)
gp.setup(servo_pin2, gp.OUT)


#p.start(1.0)

def change_pos(angle_str1, angle_str2):
    #if angle_str.isnumeric():
    angle1 = int(angle_str1)
    angle2 = int(angle_str2)
    if not (0 <= angle1 <= 180 and 0 <= angle2 <= 180):
        print('invalid input')
        return

    duty_cycle1 = (int(angle1)/18) + 2.5
    duty_cycle2 = (int(angle2)/18) + 2.5
    #p.ChangeDutyCycle(duty_cycle)
    p1 = gp.PWM(servo_pin1, 50)
    p2 = gp.PWM(servo_pin2, 50)

    p1.start(duty_cycle1)
    time.sleep(1)
    p1.stop()

    p2.start(duty_cycle2)
    time.sleep(0.6)
    #p2.stop()
    #p2.start(0)
    p2.ChangeDutyCycle(2.5)
    time.sleep(0.5)
    p2.stop()

try:
    while True:
        input_angle1, input_angle2 = input('please enter angle: ').split()
        change_pos(input_angle1, input_angle2)
        input_angle1, input_angle2 = 0, 0
except KeyboardInterrupt:
    #p.stop()
    gp.cleanup()