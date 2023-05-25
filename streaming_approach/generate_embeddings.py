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
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "bert-base-multilingual-cased",
    "distilbert-base-cased",
    "bert-base-cased",
]

def divide_chunks(list_of_sentences, elements):
    for i in range(0, len(list_of_sentences), elements):
        yield list_of_sentences[i:i + elements]

for checkpoint in checkpoints:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)
    model.to(device)
    model.eval()

    for reduced in ["", "reduced_"]:
        if not reduced:
            path_ = "../data/streams/english_german_combined_REDUCED.csv"
        else:
            path_ = "../data/static/english/reduced.csv"

        full_data = pd.read_csv(path_).iloc[:200]
        text_data = full_data["text"].tolist()



        for batch_index, sentences in tqdm(enumerate(divide_chunks(text_data, 50))):
            # batch = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
            batch = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
            # import ipdb; ipdb.set_trace()
            batch.to(device)

            with torch.no_grad():
                encoded_layers = model(**batch, return_dict=True, output_hidden_states=True)
                sentence_embeddings: torch.Tensor = torch.mean(encoded_layers[0], axis=1)
                torch.save(sentence_embeddings, f"{reduced}batch_{batch_index}.pt")

        filenames = [f"{reduced}batch_{index}.pt" for index in range(batch_index + 1)]
        print(filenames)
        combined = torch.load(filenames[0])
        for filename in filenames[1:]:
            additional = torch.load(filename)
            combined = torch.cat([combined, additional])
            del additional
        torch.save(combined, f"data/{(lambda x: 'static' if x else 'streaming')(reduced)}_embeddings_{checkpoint.replace('-', '_').replace('/', '__')}_mean_0.pt")

        for filename in filenames:
            os.remove(filename)
    del model