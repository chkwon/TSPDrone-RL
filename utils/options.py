import argparse

def str2bool(v):
    return v.lower() in ('true', '1')


def ParseParams():
    parser = argparse.ArgumentParser(description="FFEVSS Rebalancing")

    # Data generation for Training and Testing 
    parser.add_argument('--n_nodes', default=11, help="Number of nodes")
    parser.add_argument('--R', default = 150, type=int, help="Drone battery life in time units")
    parser.add_argument('--v_t', default = 1, type=int, help="Speed of truck in m/s")
    parser.add_argument('--v_d', default = 2, type=int, help="Speed of drone in m/s")
    parser.add_argument('--max_w', default = 2.5, type=float, help="Max weight a drone can carry")
    parser.add_argument('--batch_size', default= 128,type=int, help='Batch size for training')
    parser.add_argument('--n_train', default=1000000,type=int, help='# of episodes for training')
    parser.add_argument('--test_size', default= 100,type=int, help='# of instances for testing')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_path', type=str, default='trained_models/')
    parser.add_argument('--test_interval', default=200,type=int, help='test every test_interval steps')
    parser.add_argument('--save_interval', default=1000,type=int, help='save every save_interval steps')
    parser.add_argument('--log_dir', default='logs',type=str, help='folder for saving prints')
    parser.add_argument('--stdout_print', default=True, type=str2bool, help='print control')
    # Neural Network Structure 
    
    # Embedding 
    parser.add_argument('--embedding_dim', default=3,type=int, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', default=128,type=int, help='Dimension of hidden layers in Enc/Dec')
    
    # Decoder: LSTM 
    parser.add_argument('--rnn_layers', default=1, type=int, help='Number of LSTM layers in the encoder and decoder')
    parser.add_argument('--forget_bias', default=1.0,type=float, help="Forget bias for BasicLSTMCell.")
    parser.add_argument('--dropout', default=0.1, type=float, help='The dropout prob')
    
    # Attention 
    parser.add_argument('--use_tanh', type=str2bool, default=False, help='use tahn before computing probs in attention')
    parser.add_argument('--mask_logits', type=str2bool, default=True, help='mask unavailble nodes probs')
    
    # Training
    parser.add_argument('--train', default=False,type=str2bool, help="whether to do the training or not")
    parser.add_argument('--actor_net_lr', default=1e-4,type=float, help="Set the learning rate for the actor network")
    parser.add_argument('--critic_net_lr', default=1e-4,type=float, help="Set the learning rate for the critic network")
    parser.add_argument('--random_seed', default= 5,type=int, help='')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, help='Gradient clipping')
    parser.add_argument('--decode_len', default=30, type=int, help='Max number of steps per episode')
    
    # Evaluation
    parser.add_argument('--sampling', default=True,type=str2bool, help="whether to do the batch sampling or not")
    parser.add_argument('--n_samples', default=5, type=int, help='the number of samples for batch sampling')

    args, unknown = parser.parse_known_args()
    args = vars(args)
    
    for key, value in sorted(args.items()):
        print("{}: {}".format(key,value))
    
    return args 