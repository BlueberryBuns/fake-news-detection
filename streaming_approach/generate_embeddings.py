import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

dtype = torch.float
device = torch.device("mps")
test = 1
languages = "english_german"
# drift_types = ["incremental_vmild"] # , "reoccurring", "sudden"]
checkpoints = [
    # "bert-base-multilingual-cased",
    "distilbert-base-cased"
]

for checkpoint in checkpoints:
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    model = BertModel.from_pretrained(checkpoint)
    model = model.eval()

    full_data = pd.read_csv(f"../data/streams/{languages}_combined.csv", index_col=0)
    full_data["tmp_index"] = full_data.index
    # full_data.fake = full_data.fake.astype(float)
    full_data["batch_index"] = full_data.index % 300
    print(full_data.iloc[0]["text"])
    groupped_data = full_data.groupby("batch_index").agg(list)
    groupped_data = groupped_data.iloc[:5]

    for batch_index, data in tqdm(groupped_data.iterrows()):
        if batch_index == 1:
            break
        
        sentences = data["text"]
        labels = data["fake"]
        indices = data["tmp_index"]
        batch = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")["input_ids"]
        batch.to(device)

        with torch.no_grad():
            encoded_layers = model(batch, return_dict=True, output_hidden_states=True)
            sentence_embeddings: torch.Tensor = torch.mean(encoded_layers[0], axis=1)
            np_arr = sentence_embeddings.numpy()
        print(np_arr)
