from abc import abstractmethod
from typing import Optional, Type
import numpy as np
import pandas as pd
from scipy.stats import logistic
import re
from transformers import BertTokenizer, AutoTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

np.set_printoptions(threshold=np.inf)
nltk.download("stopwords")


class BaseDriftGenerator:
    def __init__(self, n_drifts: int, batch_size: int) -> None:
        self.n_drifts = n_drifts
        self.batch_size = batch_size

    def reshape_into_batches(
        self, distribution: np.ndarray, round_result: bool = False
    ) -> np.ndarray:
        n_batches = len(distribution) // self.batch_size
        batch_distribution = distribution[: n_batches * self.batch_size].reshape(
            (n_batches, self.batch_size)
        )
        return self.calculate_batch_mean(batch_distribution, round_result)

    def calculate_batch_mean(
        self, batch_distribution: np.ndarray, round_result
    ) -> np.ndarray:
        if round_result:
            return np.round(
                batch_distribution.mean(axis=1),
                np.rint(np.log10(self.batch_size)).astype(int),
            )
        return batch_distribution.mean(axis=1)

    @abstractmethod
    def generate_distribution(self, smaller_dataset_length: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def generate_drift(self, combined_dataset: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class ReoccuringDrift(BaseDriftGenerator):
    def __init__(self, n_drifts: int, batch_size: int) -> None:
        super().__init__(n_drifts, batch_size)

    def _get_batches_per_drift(self, n_batches) -> int:
        samples = int(n_batches // self.n_drifts)
        assert samples > 0
        return samples

    @staticmethod
    def binary_filter(distribution: np.ndarray):
        distribution[distribution >= 0] = 1
        distribution[distribution < 0] = 0
        return distribution

    def generate_distribution(self, smaller_dataset_length: int) -> np.ndarray:
        concept_batches_per_drift = int(
            smaller_dataset_length / self.batch_size // self.n_drifts
        )
        combined = np.concatenate(
            [np.ones(concept_batches_per_drift), np.zeros(concept_batches_per_drift)]
        )
        drift = np.repeat([combined], self.n_drifts, axis=0)
        return drift.flatten(order="C")

    def generate_drift(self, smaller_dataset_length: int) -> pd.DataFrame:
        distribution = self.generate_distribution(smaller_dataset_length)
        # batch_distibution = self.reshape_into_batches(distribution, round_result=True)
        return distribution


class GradualDrift(BaseDriftGenerator):
    DRIFT_TYPE = {
        "INF": np.iinfo(np.int32).max,
        "STEEP": 999,
        "MODERATE": 500,
        "MILD": 100,
        "vMILD": 25,
        "exMILD": 10,
    }

    def __init__(self, n_drifts: int, batch_size: int) -> None:
        super().__init__(n_drifts, batch_size)

    def sigmoid_distribution(self, period: int, slope: str):
        slope = GradualDrift.DRIFT_TYPE[slope]
        _probabilities = logistic.cdf(
            np.concatenate(
                [
                    np.linspace(
                        -slope if i % 2 else slope, slope if i % 2 else -slope, period
                    )
                    for i in range(self.n_drifts)
                ]
            )
        )
        return _probabilities

    def _get_samples_per_drift(self, smaller_dataset_length) -> int:
        samples = int(smaller_dataset_length * 2 // self.n_drifts)
        return samples

    def generate_distribution(self, smaller_dataset_length: int) -> np.ndarray:
        period = self._get_samples_per_drift(smaller_dataset_length)
        distribution = self.sigmoid_distribution(period, slope="vMILD")
        return distribution

    def generate_drift(self, smaller_dataset_length: int) -> pd.DataFrame:
        distribution = self.generate_distribution(smaller_dataset_length)
        batch_distibution = self.reshape_into_batches(distribution, round_result=True)
        return batch_distibution


class StreamGen:
    def __init__(
        self,
        concept: str,
        batch_size: int,
        n_drifts: int,
        drift_class: Type[BaseDriftGenerator],
        first_concept: str,
        vectorizer: str,
    ) -> None:
        self.concept = concept
        self.batch_size = batch_size
        self.n_drifts = n_drifts
        self.vectorizer = vectorizer
        self.first_concept = first_concept
        self._smaller_dataset_length: int
        self._smaller_dataset_concept: str
        self.tfidf_vectorizer = TfidfVectorizer()
        self.bert_vectorizer = BertTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )
        self.lm_vectorizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.dataset = self.load_dataset(
            # ["datasets/english/FinalEnglish.csv", "datasets/spanish/FinalSpanish.csv"]
            ["static/english/dataset.csv", "static/german/balanced.csv"]
        )
        self.dataset.to_csv("streams/english_german_combined.csv")
        self.drift_generator = drift_class(n_drifts=n_drifts, batch_size=batch_size)
        self.stemmer = PorterStemmer()
        self.regex = re.compile("\w+")
        self.stopwords = {
            "german": {*stopwords.words("german")},
            "english": {*stopwords.words("english")},
        }

    def _get_data_by_concept(
        self, dataframe: pd.DataFrame, concept: str
    ) -> pd.DataFrame:
        return dataframe[dataframe[self.concept] == concept]

    def _shuffle_dataset(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.sample(frac=1.0)

    def load_dataset(self, dataset_paths: list[str]) -> pd.DataFrame:
        result: Optional[pd.DataFrame] = None
        greater_dataset_concept = ""
        smaller_dataset_concept = ""
        greater_dataset_length = 0
        smaller_dataset_length = np.inf
        for dataset_path in dataset_paths:
            data = pd.read_csv(dataset_path)[["text", "fake"]]
            concept = dataset_path.split("/")[-2]
            data[self.concept] = concept
            if (size := data.shape[0]) > greater_dataset_length:
                greater_dataset_length = size
                greater_dataset_concept = concept
            if (size := data.shape[0]) <= smaller_dataset_length:
                smaller_dataset_length = size
                smaller_dataset_concept = concept
            if result is None:
                result = data
            else:
                result = pd.concat([result, data])

        self._greater_dataset_length = greater_dataset_length
        self._greater_dataset_concept = greater_dataset_concept
        self._smaller_dataset_length = smaller_dataset_length
        self._smaller_dataset_concept = smaller_dataset_concept
        return result.reset_index(drop=True)

    def _get_attachable_samples(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.iloc[self._smaller_dataset_length :]

    @property
    def concepts(self) -> list[str]:
        return list(self.dataset[self.concept].unique())

    def enhance_stream(self, batch_distribution: np.ndarray):
        enhanced_concept = (
            1 if self.first_concept == self._greater_dataset_concept else 0
        )
        extendable_indicies = np.argwhere(
            batch_distribution == enhanced_concept
        ).flatten(order="C")
        number_of_added_batches = (
            self._greater_dataset_length - self._smaller_dataset_length
        ) // self.batch_size
        selected_indicies = np.unique(
            np.sort(np.random.choice(extendable_indicies, size=self.n_drifts))
        )
        batches_per_selection = number_of_added_batches // selected_indicies.shape[0]

        print(selected_indicies)
        data_sections = []

        for i in range(selected_indicies.shape[0]):
            if i == 0:
                data_sections.append(batch_distribution[: selected_indicies[i]])

            data_sections.append(
                [enhanced_concept for _ in range(batches_per_selection)]
            )

            if i == selected_indicies.shape[0] - 1:
                data_sections.append(batch_distribution[selected_indicies[i] :])
                break

            data_sections.append(
                batch_distribution[selected_indicies[i] : selected_indicies[i + 1]]
            )

        return np.concatenate(data_sections)

    def generate_stream(self, enhance_stream: bool) -> pd.DataFrame:
        batched_data = []
        batch_distribution = self.drift_generator.generate_drift(
            self._smaller_dataset_length
        )
        if enhance_stream:
            batch_distribution = self.enhance_stream(batch_distribution)
            print(batch_distribution)
        for batch_index, distribution in enumerate(batch_distribution):
            sample_a = (
                self.dataset[self.dataset[self.concept] == self.first_concept]
                .sample(n=int(distribution * self.batch_size))
                .copy(deep=True)
            )
            sample_b = (
                self.dataset[self.dataset[self.concept] != self.first_concept]
                .sample(n=int((1 - distribution) * self.batch_size))
                .copy(deep=True)
            )
            batch = pd.concat([sample_a, sample_b])
            self.dataset = self.dataset.loc[~self.dataset.index.isin(batch)]
            batch["batch_index"] = batch_index
            batched_data.append(batch)
            # if batch_index == 5:
            #     print(sample_a)
            #     print(distribution*self.batch_size)
            #     print(sample_b)
            #     print((1-distribution)*self.batch_size)
        return pd.concat(batched_data)

    def _cleanup_pandas(self, sentence: dict) -> str:
        sentence_tok = self.regex.findall(sentence["text"])
        sentence_tok = set(map(str.lower, sentence_tok))
        sentence_tok = sentence_tok - self.stopwords[sentence[self.concept]]
        sentence_tok = list(map(PorterStemmer().stem, sentence_tok))
        sentence["text"] = " ".join(sentence_tok)
        # print(self.stopwords, sentence)
        return sentence

    def cleanup_data(self, sentences: pd.DataFrame) -> pd.DataFrame:
        batch = sentences.apply(self._cleanup_pandas, axis=1)
        return batch

    def minilm_vectorization(self, sentences: pd.Series) -> np.ndarray:
        batch = self.lm_vectorizer(
            sentences, padding=True, truncation=True, return_tensors="np"
        )

        return batch

    def bert_vectorization(self, sentences: pd.Series) -> np.ndarray:
        batch = self.bert_vectorizer(
            sentences, padding=True, truncation=True, return_tensors="np"
        )

        return batch

    def tfidf_vectorization(self, dataset: pd.DataFrame):
        dataset = self.cleanup_data(dataset)
        sentences = dataset["text"]
        result = self.tfidf_vectorizer.fit_transform(sentences)
        return result

    def vectorize(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if self.vectorizer == "tfidf":
            res = self.tfidf_vectorization(dataset)
            # print(res[0,0])

        elif self.vectorizer == "bert":
            res = self.bert_vectorization(dataset["text"].to_list())

        elif self.vectorizer == "minilm":
            res = self.minilm_vectorization(dataset["text"].to_list())

        # print(res["input_ids"])
        print(res)
        return dataset


drift_class = GradualDrift
# drift_class = ReoccuringDrift
stream_generator = StreamGen(
    concept="language",
    batch_size=100,
    n_drifts=7,
    drift_class=drift_class,
    first_concept="english",
    vectorizer="tfidf",
)
batched_data = stream_generator.generate_stream(enhance_stream=False)
# batched_data = stream_generator.vectorize(batched_data)
batched_data.reset_index().to_csv("xd.csv", index=False)
# batched_data["text"] = stream_generator.cleanup_data(batched_data["text"])
# batched_data.reset_index().to_csv("/Users/hulewicz/Private/master-thesis/datasets/combined/combined_batches_vectorized.csv", index=False)
