import glob
import os
import pandas as pd
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

dtype = torch.float
device = torch.device("mps")
test = 1
languages = "english_german"
checkpoints = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "bert-base-multilingual-cased",
    "distilbert-base-cased",
    "bert-base-cased",
    "sentence-transformers/all-MiniLM-L6-v2",
]

def divide_chunks(list_of_sentences, elements):
    for i in range(0, len(list_of_sentences), elements):
        yield list_of_sentences[i:i + elements]

for checkpoint in checkpoints:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)
    model.to(device)
    model.eval()

    full_data = pd.read_csv(f"../data/streams/{languages}_combined.csv", index_col=0)
    text_data = full_data["text"].tolist()

    for batch_index, sentences in tqdm(enumerate(divide_chunks(text_data, 100))):
        batch = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
        batch.to(device)

        with torch.no_grad():
            encoded_layers = model(**batch, return_dict=True, output_hidden_states=True)
            sentence_embeddings: torch.Tensor = torch.mean(encoded_layers[0], axis=1)
            torch.save(sentence_embeddings, f"batch_{batch_index}.pt")
    filenames = glob.glob("batch_*.pt")
    print(filenames)
    combined = torch.load(filenames[0])
    for filename in filenames[1:]:
        additional = torch.load(filename)
        combined = torch.cat([combined, additional])
    torch.save(combined, f"embeddings_{checkpoint.replace('-', '_').replace('/', '__')}_mean_0.pt")

    for filename in filenames:
        os.remove(filename)