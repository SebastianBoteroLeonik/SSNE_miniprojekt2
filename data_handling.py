import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataManager:
    def __init__(
        self, train_path="train_data.csv", test_path="test_data.csv", target="SalePrice"
    ):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.target = target

        self.cat_cols = (
            self.train_df.drop(columns=[self.target])
            .select_dtypes(include=["object", "string"])
            .columns.tolist()
        )
        self.num_cols = (
            self.train_df.drop(columns=[self.target])
            .select_dtypes(exclude=["object", "string"])
            .columns.tolist()
        )

        self.scaler = StandardScaler()
        self.cat_mappings = {}
        self.emb_dims: list[tuple[int, int]] = []
        self._process_target()
        self._process_categorical()
        self._process_numerical()

    def _process_target(self):
        self.bins = [-float("inf"), 100_000, 350_000, float("inf")]
        self.labels = [0, 1, 2]  # 0: cheap, 1: average, 2: expensive
        df = self.train_df
        df[self.target] = pd.cut(
            df[self.target], bins=self.bins, labels=self.labels
        ).astype(int)

    def _process_categorical(self):
        for col in self.cat_cols:
            unique_cats = self.train_df[col].unique().tolist()
            mapping = {val: i + 1 for i, val in enumerate(unique_cats)}
            mapping["UNK"] = 0
            self.cat_mappings[col] = mapping

            self.train_df[col] = self.train_df[col].map(mapping).fillna(0).astype(int)
            self.test_df[col] = self.test_df[col].map(mapping).fillna(0).astype(int)

            vocab_sz = len(mapping)
            self.emb_dims.append((vocab_sz, min(10, (vocab_sz + 1) // 2)))

    def _process_numerical(self):
        for col in self.num_cols:
            median_val = self.train_df[col].median()
            self.train_df[col] = self.train_df[col].fillna(median_val)
            self.test_df[col] = self.test_df[col].fillna(median_val)

        self.train_df[self.num_cols] = self.scaler.fit_transform(
            self.train_df[self.num_cols]
        )
        self.test_df[self.num_cols] = self.scaler.transform(self.test_df[self.num_cols])

    def _to_features(self, df):
        x_cat = torch.tensor(df[self.cat_cols].values, dtype=torch.long)
        x_num = torch.tensor(df[self.num_cols].values, dtype=torch.float32)
        return x_cat, x_num

    def get_train_features(self):
        return self._to_features(self.train_df)

    def get_test_features(self):
        return self._to_features(self.test_df)

    def get_train_target(self):
        return torch.tensor(self.train_df[self.target].values, dtype=torch.long)