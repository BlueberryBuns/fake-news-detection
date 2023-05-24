import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

datasets = ["english"]
checkpoints = ["bert-base-multilingual-cased"]

dataset = pd.read_csv()