from runtime import make_model, replay
from utils.parser_func import parse_args


if __name__ == '__main__':
    args = parse_args()
    if not args.init_checkpoint:
        raise ValueError('inference requires --init_checkpoint')
    replay(make_model(args).eval(), args)
