# Neural Contractive Imitation Learning
Imitating expert demonstrations with stability and predictability through learning a dynamical system has long been the center of attention. Yet, training a dynamical system to achieve stability and safety often required access to both states and its time derivative (eg, position and velocity). We propose leveraging the recently introduced Recurrent Equilibrium Networks (RENs) to achieve *state-only* imitation while providing global asymptotic stability through contraction theory. The approach hinges on differentiable ODE solvers, invertible coupling layers, and theoretical upper bounds for out-of-sample recovery.

*Note*: The repository also provides an improved and efficient implementation of continuous and discrete recurrent equilibrium networks (REN). Check the following files.

```bash
ren/ren.py # abstract REN class
ren/ren_continuous.py # continuous REN with multiple shooting
ren/ren_discrete.py # discrete ren REN multiple shooting
```


## Docker setup
To reduce the overhead of installing different simulators and environments, we already published a docker file containing all the required tools and libraries to run this codebase, and also to use Isaac Lab simulator. Our docker image is the modified version of the [Nvidia Isaac Sim docker](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html), nvcr.io/nvidia/isaac-sim:4.2.0.



### Docker image clone
### Run with/without GPU

### Nvidia drivers
### Nvidia container toolkit

## Simple CLI Usage
```bash
python train.py --device cuda:0 --total-epochs 500 --expert lasa --motion-shape Worm --num-expert-demonstration 4 --batch-size 64
```

# CLI arguments
To enable curious readers of our work to explore different setups, we provide access to a comprehensive set of model and experiment parameters through the command line interface. ```cli.py``` encompasses all these aptions, and a summary is shown below.

```bash

```
