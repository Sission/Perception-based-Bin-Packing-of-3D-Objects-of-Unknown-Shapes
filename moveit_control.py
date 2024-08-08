#!/usr/bin/env python3
import sys
import moveit_commander
import geometry_msgs.msg
from moveit_commander.conversions import pose_to_list
from franka_configurations import *
from vacuum_stop import *
from kinematics import *


def all_close(goal, actual, tolerance):
    """
  Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
  @param: goal       A list of floats, a Pose or a PoseStamped
  @param: actual     A list of floats, a Pose or a PoseStamped
  @param: tolerance  A float
  @returns: bool
  """
    # all_equal = True
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True


class RobotControl:
    def __init__(self):
        super(RobotControl, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        self.group_name = "panda_arm"

        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        self.planning_frame = self.move_group.get_planning_frame()
        self.eef_link = self.move_group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()
        self.move_group.set_max_acceleration_scaling_factor(0.5)
        self.move_group.set_max_velocity_scaling_factor(0.5)

    def go_to_joint_state(self, joints):
        joint_goal = list(joints)
        self.move_group.go(joint_goal, wait=True)
        # self.move_group.stop()
        current_joints = self.move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def info(self):
        # print self.move_group.get_current_joint_values()
        print("============ End effector link: %s" % self.eef_link)
        print("============ Current Planning Group is:", self.group_name)
        print('The end_effector pose is', self.move_group.get_current_pose())
        print('rpy is', self.move_group.get_current_rpy())

    def get_joint_pose(self):
        return self.move_group.get_current_joint_values()


class Controller:
    def __init__(self):
        # init robot
        try:
            self.robot = RobotControl()
            rospy.loginfo("Robot control launched!")
        except RuntimeError:
            rospy.logfatal("Reboot Franka Controller!")
            exit()

        self.franka_ik_solver = FrankaIK()
        self.seed_state = franka_joint_state_listener()
        self.dc = DefaultConfigurations()
        self.unsolvable = 0

    def go_to_rpy_config(self, rpy_config):
        quat_config = rpy_config2quat_config(rpy_config)
        self.go_to_quat_config(quat_config)

    def go_to_quat_config(self, quat_config):
        self.seed_state = self.inverse_kinematics(quat_config)
        self.robot.go_to_joint_state(self.seed_state)

    def inverse_kinematics(self, quat_config):
        self.seed_state = franka_joint_state_listener()
        # self.franka_ik_solver.set_auto_bound()

        solved_joints = None
        i = 0
        while type(solved_joints) != tuple:
            i += 1
            solved_joints = self.franka_ik_solver.get_solved_joint(self.seed_state, quat_config)
            # print('solved:', solved_joints)
            if i > 100:
                rospy.logfatal("Inverse Kinematics Failed!")
                # time.sleep(10)
                self.unsolvable += 1
                temp_solver = FrankaIK()
                temp_solver.set_manual_bound()
                solved_joints = temp_solver.get_solved_joint(self.seed_state, quat_config)
                raise ValueError(f"Inverse Kinematics Failed!: {quat_config}")

        self.seed_state = solved_joints
        return solved_joints

    def shutdown(self):
        error_recovery()
        self.robot.go_to_joint_state(self.dc.franka_q_init)
        rospy.loginfo('Shut down')

    def adjust_speed(self, factor):
        self.robot.move_group.set_max_acceleration_scaling_factor(factor)
        self.robot.move_group.set_max_velocity_scaling_factor(factor)
