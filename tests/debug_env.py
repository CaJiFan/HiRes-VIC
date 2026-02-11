import robosuite as suite

# Create a dummy environment
env = suite.make("Door", robots="Panda", use_camera_obs=False)

# Get the robot instance
robot = env.robots[0]

print("\n" + "="*30)
print(f"ROBOT NAME: {robot.name}")
print(f"ARM NAMES: {robot.arms}")  # <--- This is the answer
print("="*30 + "\n")