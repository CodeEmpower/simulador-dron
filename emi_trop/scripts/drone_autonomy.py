#!/usr/bin/env python
import rospy
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import CommandBool, SetMode

def arm_and_takeoff():
    rospy.init_node('drone_control', anonymous=True)
    
    # Subscribing to current drone state
    state = State()
    def state_cb(msg):
        state.connected = msg.connected
        state.armed = msg.armed
        state.mode = msg.mode

    rospy.Subscriber("/mavros/state", State, state_cb)
    
    # Wait for connection
    while not rospy.is_shutdown() and not state.connected:
        rospy.loginfo("Waiting for connection...")
        rospy.sleep(1)

    # Arming the drone
    arm_service = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
    arm_service(True)
    rospy.loginfo("Drone armed")
    
    # Wait for arming
    while not state.armed:
        rospy.loginfo("Waiting for arming...")
        rospy.sleep(1)
    
    # Setting mode to GUIDED
    set_mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)
    set_mode_service(custom_mode="GUIDED")
    rospy.loginfo("Mode set to GUIDED")

    # Create a publisher to send position commands
    position_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)
    rospy.sleep(2)  # Let the system stabilize before takeoff
    
    # Send takeoff command
    takeoff_altitude = 10  # Desired altitude in meters
    position = PoseStamped()
    position.header.frame_id = "map"  # or "local_origin", depending on your configuration
    position.pose.position.z = takeoff_altitude
    rospy.loginfo(f"Taking off to {takeoff_altitude} meters")

    for _ in range(100):  # Publish the takeoff position multiple times
        position.header.stamp = rospy.Time.now()
        position_pub.publish(position)
        rospy.sleep(0.1)

    rospy.sleep(10)  # Wait to ensure the drone reaches the desired altitude

    # Now move forward
    position.pose.position.x = 5  # Move 5 meters forward
    rospy.loginfo("Moving forward 5 meters.")
    
    for _ in range(100):  # Publish the forward position multiple times
        position.header.stamp = rospy.Time.now()
        position_pub.publish(position)
        rospy.sleep(0.1)

    rospy.sleep(5)  # Wait before starting to land

    # Start landing
    landing_altitude = 0  # Target altitude for landing
    rospy.loginfo("Starting landing sequence.")
    while position.pose.position.z > landing_altitude:
        position.pose.position.z -= 0.1  # Decrease altitude by 0.1 meters per loop
        rospy.loginfo(f"Descending to {position.pose.position.z} meters")
        position.header.stamp = rospy.Time.now()
        position_pub.publish(position)
        rospy.sleep(0.1)

    rospy.loginfo("Landed successfully!")
    rospy.spin()

if __name__ == "__main__":
    try:
        arm_and_takeoff()
    except rospy.ROSInterruptException:
        pass

