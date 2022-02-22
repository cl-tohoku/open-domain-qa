import argparse
import importlib
import logging
from pathlib import Path
from typing import Generator
import sys

from tqdm import tqdm
from omegaconf import OmegaConf


ROOT_REPOSITORY = Path(__file__).parents[1]

logging.basicConfig(
    format="%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)



if __name__ == "__main__":
    """ bash
    cd $ROOT_REPOSITORY
    python prepro/load_dataset.py JaQuAD
    """

    parser = argparse.ArgumentParser(description="To create future-aware corpus")
    parser.add_argument("data", type=str, help="key of datasets.yml")
    args = parser.parse_args()
    
    cfg_file = ROOT_REPOSITORY / "datasets.yml"
    datasets = OmegaConf.load(cfg_file)
    cfg = datasets[args.data]

    module = importlib.import_module(cfg["path"])
    data_class = getattr(module, cfg["class"])
    dataset = data_class()

    print(dataset.data)

