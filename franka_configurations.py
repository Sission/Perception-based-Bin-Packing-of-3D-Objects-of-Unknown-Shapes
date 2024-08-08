#! /usr/bin/env python3

from kinematics import *
import rospy
from franka_msgs.srv import SetForceTorqueCollisionBehavior, SetForceTorqueCollisionBehaviorRequest
from franka_msgs.msg import ErrorRecoveryActionGoal
from sensor_msgs.msg import JointState
from trac_ik_python.trac_ik import IK


def franka_joint_state_listener():
    msg = rospy.wait_for_message("/joint_states", JointState)
    return np.asarray(msg.position)


def error_recovery():
    pub = rospy.Publisher('/franka_control/error_recovery/goal', ErrorRecoveryActionGoal, queue_size=1)
    rate = rospy.Rate(1)
    goal = ErrorRecoveryActionGoal()
    goal.goal = {}
    ctrl_c = False
    while not ctrl_c:
        connections = pub.get_num_connections()
        if connections > 0:
            pub.publish(goal)
            ctrl_c = True
            rospy.loginfo('Recovered from error')
            # kinematics.format_printing('Recovered from error')
        else:
            rate.sleep()


def set_force():
    rospy.wait_for_service('/franka_control/set_force_torque_collision_behavior')
    ftcb_srv = rospy.ServiceProxy('/franka_control/set_force_torque_collision_behavior',
                                  SetForceTorqueCollisionBehavior)
    ftcb_msg = SetForceTorqueCollisionBehaviorRequest()
    print(ftcb_msg, '---', ftcb_srv)
    return ftcb_msg, ftcb_srv


def move_defaults():
    msg, srv = set_force()
    msg.upper_torque_thresholds_nominal = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
    msg.upper_force_thresholds_nominal = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
    print('1111')

    res = srv.call(msg).success
    print(res, '?')
    if not res:
        print('Failed to set Force/Torque Collision Behaviour Thresholds')

    else:
        print('Successfully set Move-Mode Force/Torque Collision Behaviour Thresholds')


def test_forceset():
    rospy.loginfo('Waiting for '
                  'franka_control/set_force_torque_collision_behavior')
    set_collision_behavior = rospy.ServiceProxy(
        'franka_control/set_force_torque_collision_behavior', SetForceTorqueCollisionBehavior)
    set_collision_behavior.wait_for_service()
    torques = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
    forces = [20.0, 20.0, 30.0, 25.0, 25.0, 25.0]
    msg, srv = set_force()

    print(srv.call(msg))
    # set_collision_behavior(lower_torque_thresholds_nominal=torques,
    #                        upper_torque_thresholds_nominal=torques,
    #                        lower_force_thresholds_nominal=forces,
    #                        upper_force_thresholds_nominal=forces)


class DefaultConfigurations:
    def __init__(self):
        # Franka Emika Initial(Home)
        self.franka_q_init = np.array([0, -np.pi / 4, 0, -3 / 4 * np.pi, 0, np.pi / 2, 0])
        self.franka_q_checkpoint = np.array([0, -np.pi / 4, 0, - 3.1 / 4 * np.pi, 0, np.pi / 2, 0])
        self.franka_q_test = np.array([-np.pi / 4, -np.pi / 4, 0, -3 / 4 * np.pi, 0, np.pi / 2, 0])

        # Franka Panda Screw Axis
        self.franka_S = np.array([[0, 0, 0, 0, 0, 0, 0],
                                  [0, 1, 0, -1, 0, -1, 0],
                                  [1, 0, 1, 0, 1, 0, -1],
                                  [0, -0.333, 0, 0.649, 0, 1.033, 0],
                                  [0, 0, 0, 0, 0, 0, 0.088],
                                  [0, 0, 0, -0.0825, 0, 0, 0]])
        # Franka Panda Home Matrix
        self.franka_M = np.array([[1, 0, 0, 0.088],
                                  [0, -1, 0, 0],
                                  [0, 0, -1, 0.926],
                                  [0, 0, 0, 1]])


class FrankaIK(IK):
    def __init__(self):
        super(FrankaIK, self).__init__('panda_link0', 'panda_link8')
        self.manual_upper_bound = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        self.manual_lower_bound = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        #
        # self.auto_upper_bound = [0.1, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 0.00001]
        # self.auto_lower_bound = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -0.00001]

        self.auto_upper_bound = [1, 1, 2.8973, -0.0698, 2.8973, 3.7525, 0.00001]
        self.auto_lower_bound = [-1, -1, -2.8973, -3.0718, -2.8973, -0.0175, -0.00001]

    # [-0.00023346695941829827, -0.7856274022805063, 0.00046524589335170097, -2.3557260509290194,
    #  -0.0007756932397476502,  1.5713376433849333, -0.0004680079749260815]

    # -1.0921873771098622, 0.6754607260191572, 0.46879893207131773, -2.1923135524214357, \
    # -0.7747042324111841, 3.513385278532407, -0.00046023904177774135

    def set_auto_bound(self):
        self.set_joint_limits(self.auto_lower_bound, self.auto_upper_bound)

    def set_manual_bound(self):
        self.set_joint_limits(self.manual_lower_bound, self.manual_upper_bound)

    def get_solved_joint(self, q_init, config):
        solved_joints = self.get_ik(q_init, config[4], config[5], config[6], config[0],
                                    config[1], config[2], config[3],
                                    bx=1e-5, by=1e-5, bz=1e-5,
                                    brx=1e-3, bry=1e-3, brz=1e-3)
        return solved_joints


class WorldMap:
    def __init__(self, uncertainty, angle, area, target):
        self.peg_structure_height = 0.05
        peg_height_set = [0.05, 0.0196, 0.0157]
        rotate_set = [[0, 0, 0], [0, -np.pi / 9, 0], [-np.pi / 12, 0, 0]]

        self.peg_height = peg_height_set[angle]
        self.rotate = rotate_set[angle]

        area_set = [[0.3, -0.3, 0.04 + self.peg_height], [0.35, 0.299, 0.041 + self.peg_height]]

        self.area = area_set[area]
        self.end_effector_height = 0.167

        self.picking_hole_world = rpy_config2matrix([*[0, 0, 0], *self.area])
        self.transformation = rpy_config2matrix([np.pi, 0, 0, 0, 0, 0])
        self.orientation = rpy_config2matrix([*self.rotate, *[0, 0, 0]])
        self.end_effector = rpy_config2matrix([0, 0, 0, 0, 0, -self.end_effector_height])

        # Picking peg structure surface with respect to Franka
        self.hole_add_peg_matrix = np.linalg.multi_dot([self.picking_hole_world, self.transformation, self.orientation])

        # Franka EE configuration for suction contact
        self.suction_contact_matrix = np.matmul(self.hole_add_peg_matrix, self.end_effector)
        self.suction_contact_rpy_config = matrix2rpy_config(self.suction_contact_matrix)

        # Franka EE configuration for suction ready
        self.suction_ready_rpy_config = self.suction_contact_rpy_config.copy()
        self.suction_ready_rpy_config[-1] += 0.065

        # Target area:
        if target == "A1":
            self.target_world_pose = [0, -np.pi / 6, 0, 0.5522, 0.273, 0.1555]
        elif target == "A2":
            self.target_world_pose = [0, 0, 0, 0.631, 0.0015, 0.0865]
        elif target == "A3":
            self.target_world_pose = [0, -np.pi / 4, 0, 0.546, -0.263, 0.105]

        self.placing_hole_world_rpy_config = self.target_world_pose.copy()
        self.placing_hole_world_rpy_config[-1] += self.peg_height + self.peg_structure_height
        self.placing_hole_world = rpy_config2matrix(self.placing_hole_world_rpy_config)
        # Placing peg structure surface with respect to Franka
        # if target == "A1":
        #     self.placing_hole_world = rpy_config2matrix(
        #         [0, -np.pi / 6, 0, 0.55, 0.2725, 0.154 + self.peg_height + self.peg_structure_height])
        # elif target == "A2":
        #     self.placing_hole_world = rpy_config2matrix(
        #         [0, 0, 0, 0.631, 0.0015, 0.0865 + self.peg_height + self.peg_structure_height])
        # elif target == "A3":
        #     self.placing_hole_world = rpy_config2matrix(
        #         [0, -np.pi/4, 0, 0.546, -0.263, 0.105 + self.peg_height + self.peg_structure_height])
        self.placing_hole_matrix = np.linalg.multi_dot([self.placing_hole_world, self.transformation, self.orientation])

        # Franka EE configuration for correct contact
        self.target_contact_matrix = np.matmul(self.placing_hole_matrix, self.end_effector)
        self.target_contact_rpy_config = matrix2rpy_config(self.target_contact_matrix)

        # Franka EE configuration for correct ready
        self.target_ready_rpy_config = self.target_contact_rpy_config.copy()
        if target == "A2":
            self.target_ready_rpy_config[-1] += 0.05
        else:
            w = translation(self.target_ready_rpy_config, -0.05)
            self.target_ready_rpy_config = quat_config2rpy_config(translation(self.target_ready_rpy_config, -0.05))

        # Uncertainty Simulation
        uncertainty_rpy = [self.target_world_pose[0] + uncertainty[0], self.target_world_pose[1] + uncertainty[1],
                           self.target_world_pose[2] + uncertainty[2]]
        self.placing_hole_world_with_uncertainty_config = rpy_config2matrix([*uncertainty_rpy,
                                                                             *self.target_world_pose[3:6]])
        print([*uncertainty_rpy, *self.target_world_pose[3:6]])
        # print(self.placing_hole_world_with_uncertainty_config)
        self.placing_hole_world_with_uncertainty = \
            np.matmul(self.placing_hole_world_with_uncertainty_config,
                      rpy_config2matrix([0, 0, 0, 0, 0, self.peg_height + self.peg_structure_height]))
        temp = [*uncertainty_rpy, *self.target_world_pose[3:6]]
        temp[-1] += self.peg_height + self.peg_structure_height
        self.placing_hole_world_with_uncertainty = rpy_config2matrix(temp)

        self.placing_hole_matrix_with_uncertainty = np.linalg.multi_dot(
            [self.placing_hole_world_with_uncertainty, self.transformation, self.orientation])

        # Franka EE configuration for target contact (nominal starting configuration)
        self.nominal_contact_matrix = np.matmul(self.placing_hole_matrix_with_uncertainty, self.end_effector)
        self.nominal_contact_rpy_config = matrix2rpy_config(self.nominal_contact_matrix)
        # Franka EE configuration for target ready (nominal ending configuration)
        self.nominal_ready_matrix = np.linalg.multi_dot([self.placing_hole_world_with_uncertainty,
                                                         rpy_config2matrix([0, 0, 0, 0, 0, 0.05]),
                                                         self.transformation, self.orientation, self.end_effector])
        self.nominal_ready_rpy_config = matrix2rpy_config(self.nominal_ready_matrix)


def main():
    rospy.init_node("franka_configurations")
    rospy.Rate(100)
    print('-------------------')


def test():
    k = WorldMap(uncertainty=[-0.02, -0.16, -0.11], angle=0, area=0, target='A2')


if __name__ == '__main__':
    test()
