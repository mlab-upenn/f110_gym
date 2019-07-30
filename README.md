# f110_gym
## Description
Makes the F110 an OpenAI Gym Environment. Also allows distributed computation by offloading batches to a server, and recieving a neural network back.
Currently can do a bunch of cool things, check out ```f110_core.py``` & algorithms directory to get a better idea. 

## Installation
```bash
cd ~/catkin_ws/src
git clone https://github.com/mlab-upenn/f110_gym.git f110_gym
catkin_create_pkg f110_gym
```
Additionally, add ```f110_gym/src``` to your PYTHONPATH 
