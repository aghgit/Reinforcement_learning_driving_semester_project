from beamngpy import BeamNGpy, Scenario, Vehicle
import time

# Step 1: Connect to BeamNG.tech instance
beamng = BeamNGpy(
    host='localhost',
    port=64256,
    home='D:\\1-Prog\\_PROGRAMMATION\\Project Management\\Self_Driving_Car_RL\\BeamNG.tech.v0.35.5.0'

)

beamng.open()

# Step 2: Create scenario on your custom map
scenario = Scenario('driver_training', 'ppo_test_scenario')

# Step 3: Add vehicle to scenario
vehicle = Vehicle('ego_vehicle', model='etk800', licence='PPO')
scenario.add_vehicle(vehicle, pos=(0, 0, 0), rot=(0, 0, 0))  # May need better coords

# Step 4: Generate scenario files
scenario.make(beamng)

# Step 5: Load and run
beamng.load_scenario(scenario)
beamng.start_scenario()

# Step 6: Optional delay
time.sleep(5)

# Step 7: Clean exit
beamng.close()
