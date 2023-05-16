import logging
from utils.arg import parse_args
args = parse_args()
def log():
    '''写日志文档'''
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger("Model")
    # logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')
    logger.info(f'loading pretrained model and tokenizer of {args.model_type} ...')
    return logger
logger = log()
