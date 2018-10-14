#!/bin/bash

# Start Docker if is not already running
sudo systemctl start docker

# Command to open a session of FEniCS using Docker ...
# Files stored in the /home/fenics/shared folder will be saved on the current directory of the host machine
sudo docker run -ti -p 127.0.0.1:8000:8000 -v $(pwd):/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable:current
