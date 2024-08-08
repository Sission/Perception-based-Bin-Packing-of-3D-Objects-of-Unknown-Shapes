#!/usr/bin/env python3

# import sys
import rospy
from cobot_pump_ros.srv import *
import time


class VacuumController:
    def __init__(self):
        rospy.wait_for_service('startPump')
        self.start_pump = rospy.ServiceProxy('startPump', startPump)
        self.drop_item = rospy.ServiceProxy('dropItem', dropItem)
        self.check_item_attached = rospy.ServiceProxy('checkItemAttached', checkItemAttached)
        self.stop_pump = rospy.ServiceProxy('stopPump', stopPump)
        self.vacuumStrength = 600
        self.timeout_ms = 1000

    def activate(self, timeout):
        rospy.loginfo('Activating pump now')
        # timeout_s -> timeout_ms
        response = self.start_pump(self.vacuumStrength, timeout * 1000)
        rospy.loginfo(response.vacuumSuccess)
        activation_status = response.vacuumSuccess
        rospy.loginfo(f"status of pump is {activation_status}")
        if not activation_status:
            self.shutdown()
        else:
            self.check_attachment()

    def check_attachment(self):
        response = self.check_item_attached()
        rospy.loginfo("Attach item now")
        attachment_status = response.itemAttached
        if attachment_status:
            rospy.loginfo("Item still attached")
            return True
        else:
            rospy.loginfo("No item attached")
            return False

    def release(self, timeout):
        # timeout_s -> timeout_ms
        response = self.drop_item(timeout * 1000)
        rospy.loginfo("dropping item now")
        drop_status = response.success
        if drop_status:
            rospy.loginfo("Item dropped")
            return True
        else:
            rospy.loginfo("Item dropping failed")
            return False

    def shutdown(self):
        response = self.stop_pump()
        rospy.loginfo("Shutting down pump now")
        stop_status = response.success
        if stop_status:
            rospy.loginfo("Pump shut down successfully")
            return True
        else:
            rospy.loginfo("Pump shut down successfully failed")
            return False


if __name__ == "__main__":
    rospy.init_node("vacuum_activation")
    controller = VacuumController()
    controller.activate(500)
