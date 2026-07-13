import argparse

def print_args(args):
    length = 30
    print(f"{'Arguments':-^{length*3}}")
    for key, val in sorted(vars(args).items()):
        print(f"{key:<{15}}: {val}")
    print("-"*length*3)

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

    group.add_argument(
        '--batch-size', type=int, default=None,
        dest='batch_size',
        help='global batch size'
    )

    group.add_argument(
        '--eval-batch-size', type=int, default=None,
        dest='eval_batch_size',
        help='batch size for eval'
    )

    group.add_argument(
        '--train-iters', type=int, default=None,
        dest='train_iters',
        help='total number of training steps'
    )

    group.add_argument(
        '--save', type=str, default='./ckpt.pt',
        help='path to where the checkpoints will be saved to'
    )

    group.add_argument(
        '--load', type=str, default='./ckpt.pt',
        help='path to where the checkpoints will be loaded from'
    )

    group.add_argument(
        '--save-interval', type=int, default=None,
        dest='save_interval',
        help='frequency for saving the ckpts'
    )

    group.add_argument(
        '--train-data-path', type=str, default=None,
        dest='train_data_path',
        help='path to where the tokenized corpus is (should have extension .npy)'
    )

    group.add_argument(
        '--valid-data-path', type=str, default=None,
        dest='valid_data_path',
        help='path to where the tokenized corpus is (should have extension .npy)'
    )

    group.add_argument(
        '--eval-interval', type=int, default=None,
        dest='eval_interval',
        help='evaluation frequency'
    )

    return parser

def _add_optimizer_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('(adamw) optimizer configs')

    group.add_argument(
        '--lr', type=float, default=1e-4,
        help='learning rate'
    )

    group.add_argument(
        '--adam-beta1', type=float, default=0.9,
        dest='adam_beta1',
        help='beta1 for adam optimizer'
    )

    group.add_argument(
        '--adam-beta2', type=float, default=0.95,
        dest='adam_beta2',
        help='beta2 for adam optimizer'
    )

    group.add_argument(
        '--adam-eps', type=float, default=1e-8,
        dest='adam_eps',
        help='eps in adam optimizer'
    )

    group.add_argument(
        '--weight-decay', type=float, default=0.1,
        dest='weight_decay',
        help='weight decay in adamw optimizer'
    )

    group.add_argument(
        '--clip-grad', type=float, default=1.0,
        dest='clip_grad',
        help='grad clipping'
    )    

    group.add_argument(
        '--min-lr', type=float, default=1e-5,
        dest='min_lr',
        help='minimum learning rate in cos weight decay'
    )

    group.add_argument(
        '--warmup-iters', type=int, default=None,
        dest='warmup_iters',
        help='number of iterations for warming up learning rate linearly'
    )

    return parser

def _add_model_arch_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('model configs')

    group.add_argument(
        '--vocab-size', type=int, default=None, 
        dest='vocab_size',
        help='vocab size'
    )

    group.add_argument(
        '--context-length', type=int, default=None,
        dest='context_length',
        help='sequence length'
    )

    group.add_argument(
        '--num-layers', type=int, default=None,
        dest='num_layers',
        help='number of decoder layers in the model'
    )

    group.add_argument(
        '--num-heads', type=int, default=None,
        dest='num_heads',
        help='number of attention heads'
    )

    group.add_argument(
        '--d-model', type=int, default=None,
        dest='d_model',
        help='model dim'
    )

    group.add_argument(
        '--d-ff', type=int, default=None,
        dest='d_ff',
        help='dim of the FFN hidden layer in SwiGLU'
    )

    group.add_argument(
        '--rope-theta', type=float, default=None,
        dest='rope_theta',
        help='theta in RoPE'
    )

    return parser
