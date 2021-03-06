#!/usr/bin/env python3

import time
import RPi.GPIO as gp

servo_pin = 12

gp.setmode(gp.BCM)

gp.setup(servo_pin, gp.OUT)


#p.start(1.0)

def get_dc(angle):
    return ((int(angle)/18) + 2.5)

def change_pos(angle_str):
    #if angle_str.isnumeric():
    if True:
        angle = int(angle_str)
        if not (0 <= angle <= 180):
            print('invalid input')
            return 1
    else:
        print('invalid input')
        return 1

    #duty_cycle = (int(angle)/18) + 2.5
    duty_cycle = get_dc(angle)
    #p.ChangeDutyCycle(duty_cycle)
    p = gp.PWM(servo_pin, 50)
    p.start(duty_cycle)
    time.sleep(0.15)
    p.ChangeDutyCycle(get_dc(180))
    time.sleep(1)
    p.stop()

try:
   while True:
        input_angle = input('please enter angle: ')
        change_pos(input_angle)
        input_angle = 0
except KeyboardInterrupt:
    #p.stop()
    gp.cleanup()