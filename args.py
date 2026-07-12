import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Arguments',
                                     allow_abbrev=False)

    parser = add_arguments(parser)
    args = parser.parse_args()

    return args

def add_arguments(parser: argparse.ArgumentParser):
    parser = _add_model_arch_args(parser)
    parser = _add_optimizer_args(parser)
    parser = _add_training_args(parser)
    return parser

def _add_training_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('training configs')

    group.add_arguments(
        '--batch-size', type=int, default=None,
        dest='batch_size',
        help='global batch size'
    )

    group.add_arguments(
        '--train-iters', type=int, default=None,
        dest='train_iters'
        help='total number of training steps'
    )

    group.add_arguments(
        '--save', type=str, default='./ckpt.pt',
        help='path to where the checkpoints will be saved to'
    )

    group.add_arguments(
        '--load', type=str, default='./ckpt.pt',
        help='path to where the checkpoints will be loaded from'
    )

    group.add_arguments(
        '--save-interval', type=int, default=None,
        dest='save_interval',
        help='frequency for saving the ckpts'
    )

    group.add_arguments(
        '--data-path', type=str, default=None,
        dest='data_path',
        help='path to where the corps are'
    )

    return parser

def _add_optimizer_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('(adamw) optimizer configs')

    group.add_arguments(
        '--lr', type=float, default=1e-4,
        help='learning rate'
    )

    group.add_arguments(
        '--adam-beta1', type=float, default=0.9,
        dest='adam_beta1',
        help='beta1 for adam optimizer'
    )

    group.add_arguments(
        '--adam-beta2', type=float, default=0.95,
        dest='adam_beta2',
        help='beta2 for adam optimizer'
    )

    group.add_arguments(
        '--adam-eps', type=float, default=1e-8,
        dest='adam_eps',
        help='eps in adam optimizer'
    )

    group.add_arguments(
        '--weight-decay', type=float, default=0.1,
        dest='weight_decay',
        help='weight decay in adamw optimizer'
    )

    group.add_arguments(
        '--clip-grad', type=float, default=1.0,
        dest='clip_grad',
        help='grad clipping'
    )    

    group.add_arguments(
        '--min-lr', type=float, default=1e-5,
        dest='min_lr',
        help='minimum learning rate in cos weight decay'
    )

    group.add_arguments(
        '--warmup-iters', type=int, default=None,
        dest='warmup_iters',
        help='number of iterations for warming up learning rate linearly'
    )

    return parser

def _add_model_arch_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('model configs')

    group.add_arguments(
        '--vocab-size', type=int, default=None, 
        dest='vocab_size'
        help='vocab size'
    )

    group.add_arguments(
        '--context-length', type=int, default=None,
        dest='context_length',
        help='sequence length'
    )

    group.add_arguments(
        '--num-layers', type=int, default=None,
        dest='num_layers',
        help='number of decoder layers in the model'
    )

    group.add_arguments(
        '--d-model', type=int, default=None,
        dest='d_model',
        help='model dim'
    )

    group.add_arguments(
        '--d-ff', type=int, default=None,
        dest='d_ff',
        help='dim of the FFN hidden layer in SwiGLU'
    )

    group.add_arguments(
        '--rope-theta', type=float, default=None,
        dest='rope_theta',
        help='theta in RoPE'
    )

    return parser
