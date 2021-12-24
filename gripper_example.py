import robotiq_gripper
import time
import sys

ip = "192.168.8.113"     # actual ip of the UR robot

def log_info(gripper):
    print(f"Pos: {str(gripper.get_current_position()): >3}  "
          f"Open: {gripper.is_open(): <2}  "
          f"Closed: {gripper.is_closed(): <2}  ")

if len(sys.argv) != 2:
    print('False')
    sys.exit()

print("Creating gripper...")
gripper = robotiq_gripper.RobotiqGripper()
print("Connecting to gripper...")
gripper.connect(ip, 63352)
print("Activating gripper...")
gripper.activate(auto_calibrate=False)

print("Testing gripper...")
if sys.argv[1] == '0':
    gripper.move_and_wait_for_pos(255, 255, 255)
    log_info(gripper)
if sys.argv[1] == '1':
    gripper.move_and_wait_for_pos(120, 255, 255)
    log_info(gripper)

print(gripper.get_current_position())
