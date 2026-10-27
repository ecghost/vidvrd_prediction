from runtime import make_model, evaluate_splits
from utils.parser_func import parse_args
from utils.run_logging import setup_logger


if __name__ == '__main__':
    args = parse_args()
    if not args.init_checkpoint:
        raise ValueError('inference requires --init_checkpoint')
    logger = setup_logger(args, inference=True)
    evaluate_splits(make_model(args).eval(), args, logger)
