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

Training data is generated on the fly with the batch size and node numbers specified in `utils/options.py`. If test data is not given in the data folder, the test data will be generated randomly as well.

### Training

For training TSPD, just run the following line. Any other training parameters can also be set in `utils/options.py` such as the number of nodes, batch size, the number of epochs, decode lengths, etc. 
```bash
python main.py --train=True
```

### Evaluation
 To perform only inference, please set `train` to `False` in `utils/options.py` or just run:
```bash
python main.py --train=False
```
By default, the greedy decoding will run. 

#### Sampling
To run batch sampling, please set `sampling` to `True` and specify the number of samples `n_samples` in `utils/options.py`. 

The results of both greedy and batch sampling decoding will be stored in the `results` folder. 

### Example TSPD solution
A sample solution of TSPD for 11 nodes is depicted below:
![example-single-1](https://user-images.githubusercontent.com/25514362/93505944-8c1cfd80-f8e9-11ea-81af-ae5d10f5eeaf.png)



## Acknowledgements
This repository heavily benefited from the following repositories:
- https://github.com/wouterkool/attention-learn-to-route
- https://github.com/OptMLGroup/VRP-RL
