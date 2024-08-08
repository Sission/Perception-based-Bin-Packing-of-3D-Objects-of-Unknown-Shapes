#!/usr/bin/env python3

from vacuum_control import *


def deactivation(timeout):
    time.sleep(1)
    stop_controller = VacuumController()
    stop_controller.release(timeout)
    stop_controller.shutdown()


def main():
    rospy.init_node("vacuum_deactivation")
    deactivation(5)


if __name__ == '__main__':
    main()
