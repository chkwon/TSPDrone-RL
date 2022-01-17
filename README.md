# A Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drone

This repository contains code for deep reinforcement learning to solve the Traveling Salesman Problem with Drone (TSPD). For details, please see our paper [A Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drone](https://arxiv.org/abs/2112.12545). If this code is useful for your work, please cite our paper:

```
@misc{bogyrbayeva2021deep,
      title={A Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drone}, 
      author={Aigerim Bogyrbayeva and Taehyun Yoon and Hanbum Ko and Sungbin Lim and Hyokun Yun and Changhyun Kwon},
      year={2021},
      eprint={2112.12545},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
``` 

For the optimization heuristic algorithms used in the paper, please see [TSPDrone.jl](https://github.com/chkwon/TSPDrone.jl).


## Dependencies

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7


## Usage

### Generating data

Training data is generated on the fly with the batch size and node numbers specified in `/utils/options.py`. If test data is not given in the data folder, the test data will be generated randomly as well.

### Training

For training TSPD, just run the following line. Any other training parameters can also be set in `/utils/options.py` such as the number of nodes, batch size, the number of epochs, decode lengths, etc. 
```bash
python main.py --train=True
```
The trained weight files will be saved in the `/trained_models` directory.

Pre-trained weights files for random data as described in the paper are located in the `/trained_models/pretrained_random_data` directory. To use these weight files, copy the two `.pkl` files to the `/trained_models` directory.


### Evaluation
 To perform only inference, please set `train` to `False` in `/utils/options.py` or just run:
```bash
python main.py --train=False
```
By default, the greedy decoding will run. 

### Sampling
To run batch sampling, please set `sampling` to `True` and specify the number of samples `n_samples` in `/utils/options.py`. 

The results of both greedy and batch sampling decoding will be stored in the `results` folder. 




## Test Instances

The `/data` directory includes random test instances used in the paper for `n=11, 15, 20, 50, 100`.
Each file includes 100 instances. 

Each row represents an instance, in the form of 
```
x_1 y_1 x_2 y_2 ... x_n y_n
```
where `x_i, y_i` represents the coordinate of customer `i`, and `x_n y_n` represents the depot.



## Example TSPD solution
A sample solution of TSPD for 11 nodes is depicted below:
![](/images/optimal-n11-6-2.jpg)



## Acknowledgements
This repository heavily benefited from the following repositories:
- https://github.com/wouterkool/attention-learn-to-route
- https://github.com/OptMLGroup/VRP-RL
