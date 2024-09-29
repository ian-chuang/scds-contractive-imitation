# SCDS: State-only framework for learning Contractive Dynamical System policies through imitation
Imitating expert demonstrations with stability and predictability through learning a dynamical system has long been the center of attention. Yet, training a dynamical system to achieve stability and safety often required access to both states and its time derivative (eg, position and velocity). We propose leveraging the recently introduced Recurrent Equilibrium Networks (RENs) to achieve *state-only* imitation while providing global asymptotic stability through contraction theory. The approach hinges on differentiable ODE solvers, invertible coupling layers, and theoretical upper bounds for out-of-sample recovery.

**The corresponding manuscript is under review at ICLR 2025.**

## Docker setup
To reduce the overhead of installing different simulators and environments, we already published a docker file containing all the required tools and libraries to run this codebase, and also to use Isaac Lab simulator. Our docker image is the modified version of the [Nvidia Isaac Sim docker](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html), nvcr.io/nvidia/isaac-sim:4.2.0.


### Pull docker image from DockerHub
**NOTE: Removed for anonymity, available as part of supplementary materials**

### Download the DockerImage


## Basic usage

### Training
```bash
# LASA expert data
python train.py --device cuda:0 --total-epochs 500 --expert lasa --motion-shape Worm --num-expert-demonstration 1

# Robomimic expert data
    python train.py --expert "robomimic" --motion-shape "lift"  --dim-in 3 --dim-out 3 --device cuda:0  --total-epochs 500 --bijection --num-bijection-layers 8 --crate-lb 12.0  --num-expert-trajectories 1 --horizon 20 --loss dtw &

```

### Testing and plots
```bash
python test.py --load-model data/trained_policy/dren-lasa-Worm-h50-x64-e15000-b16-cr1.0-e1-s50-a0-t11-0839/best_model.pth
```

## Command line arguments
To enable curious readers of our work to explore different setups, we provide access to a comprehensive set of model and experiment parameters through the command line interface. ```cli.py``` encompasses all these options, and a summary of key items is shown below.

## Train parameters
Use the following command to get the full list of possible arguments.
```bash
python train.py --help
```

The key arguments are summarized below.
```bash
  # model
  --model-type MODEL_TYPE
                        Choose a model between "continuous" and "discrete" for the underlying REN.
  --device DEVICE       Device to run the computations on, "cpu" or "cuda:0". Default is "cuda:0" if available, otherwise "cpu".
  --horizon HORIZON     Horizon value for the computation. Default is 50.
  --dim-x DIM_X         Dimension x. Default is 32.

  --bijection           Use bijection net before projecting the output.
  --num-bijection-layers NUM_BIJECTION_LAYERS
                        Number of hidden layers in the coupling layer design and blocks. Default is 2.

  # training
  --total-epochs TOTAL_EPOCHS
                        Total number of epochs for training. Default is 200.
  --crate-lb CRATE_LB   Lower bound for the contraction rate. Defaults to 0.0 for continuous and 1 for discrete.
  --loss LOSS           Training loss to be selected between "dtw" and "mse". Default is "mse".

  # dataset
  --expert EXPERT       Expert type among ["lasa", "robomimic"]. Default is "lasa".
  --motion-shape MOTION_SHAPE
                        Motion shape in LASA or Robomimic dataset. Choose from ["Angle", "CShape", "GShape", "Sine", "Snake", "Worm", etc] for LASA and ["lift", "can",
                        "transport", "square"] for robomimic.
  --dataset-key DATASET_KEY
                        Robomimic dataset keys in ["eef_pos", "eef_pos_ori", "joint_pos", "joint_pos_vel"]
```

## Test parameters
A summary of key parameters may be found in the following. We use the same CLI interface for train and test to remain consistent, but that means there are some train parameters which remain unused during test and vice versa.

```bash
  --num-test-rollouts NUM_TEST_ROLLOUTS
                        Number of test rollouts for plots.
  --ic-test-std IC_TEST_STD
                        Initial condition std during test and plotting phase.
  --load-model LOAD_MODEL
                        If it is not set to None, a pretrained model will be loaded instead of training.
  --legends             Add legend to the plots.
  --new-ic-test         Load the saved initial conditions for consistency with other baselines.
```


## REN implementations
*Note*: The repository also provides an improved and efficient implementation of continuous and discrete recurrent equilibrium networks (REN). Check the following files.

```bash
ren/ren.py # abstract REN class
ren/ren_continuous.py # continuous REN with multiple shooting
ren/ren_discrete.py # discrete ren REN multiple shooting
```

These implementations are built upon:
* [https://github.com/DecodEPFL/NodeREN](https://github.com/DecodEPFL/NodeREN)
* [https://github.com/DecodEPFL/perf-boost-base](https://github.com/DecodEPFL/perf-boost-base)


## Contribution guide
We welcome contributions that improve or extend SCDS in any way. Please refer to [Contribution.md](Contribution.md) for more information on this.

## Corresponding authors
**NOTE: Removed for anonymity, available as part of supplementary materials**
