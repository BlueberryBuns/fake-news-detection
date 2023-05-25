import torch

import pandas as pd
from transformers import AutoModel, BertModel, BertTokenizerFast
import torch
from tqdm import tqdm
import torch._dynamo

dtype = torch.float
device = torch.device("mps")


checkpoint = "bert-base-cased"
full_data = pd.read_csv(f"../data/static/english/reduced.csv")
print(full_data.iloc[0]["text"])
data = full_data[:1]
sentences = data["text"].to_list()
labels = data["fake"].to_list()

def divide_chunks(list_of_sentences, elements):
    for i in range(0, len(list_of_sentences), elements):
        yield list_of_sentences[i:i + elements]
tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
# model = BertModel.from_pretrained(checkpoint)
# model.to(device)
# model.eval()
model = BertModel.from_pretrained(checkpoint)
# model.to(device)
model.eval()


for i, chunk in tqdm(enumerate(divide_chunks(list_of_sentences=sentences, elements=100))):
    batch = tokenizer(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt")
    # batch.to(device)
    with torch.no_grad():
        encoded_layers = model(**batch, return_dict=True, output_hidden_states=True)
        import ipdb; ipdb.set_trace()
        sentence_embeddings: torch.Tensor = torch.mean(encoded_layers[0], axis=1)
        # # np_arr = sentence_embeddings.numpy()
        torch.save(sentence_embeddings, f"tensor_{i}.pt")
    # print(np_arr)