import os
import json
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
import random

class CLIPEmbeddingDataset(Dataset):
    synsetMapping = {
        "02691156": "airplane",
        "02747177": "trash bin",
        "02773838": "bag",
        "02801938": "basket",
        "02808440": "bathtub",
        "02818832": "bed",
        "02828884": "bench",
        "02843684": "birdhouse",
        "02871439": "bookshelf",
        "02876657": "bottle",
        "02880940": "bowl",
        "02924116": "bus",
        "02933112": "cabinet",
        "02942699": "camera",
        "02946921": "can",
        "02954340": "cap",
        "02958343": "car",
        "02992529": "cellphone",
        "03001627": "chair",
        "03046257": "clock",
        "03085013": "keyboard",
        "03207941": "dishwasher",
        "03211117": "display",
        "03261776": "earphone",
        "03325088": "faucet",
        "03337140": "file cabinet",
        "03467517": "guitar",
        "03513137": "helmet",
        "03593526": "jar",
        "03624134": "knife",
        "03636649": "lamp",
        "03642806": "laptop",
        "03691459": "loudspeaker",
        "03710193": "mailbox",
        "03759954": "microphone",
        "03761084": "microwaves",
        "03790512": "motorbike",
        "03797390": "mug",
        "03928116": "piano",
        "03938244": "pillow",
        "03948459": "pistol",
        "03991062": "flowerpot",
        "04004475": "printer",
        "04074963": "remote",
        "04090263": "rifle",
        "04099429": "rocket",
        "04225987": "skateboard",
        "04256520": "sofa",
        "04330267": "stove",
        "04379243": "table",
        "04401088": "telephone",
        "04460130": "tower",
        "04468005": "train",
        "04530566": "watercraft",
        "04554684": "washer",
    }

    def __init__(self, rootDir, csvFile, categories=None, split='clip-vit-base-patch32', synsetJson=None):
        self.rootDir = os.path.join(rootDir, split)
        self.csvFile = csvFile
        self.categories = [cat.lstrip('0') for cat in categories] if categories else None
        self.data = []

        self.data_dict = {}

        if synsetJson:
            with open(synsetJson, "r") as f:
                self.synsetToCategory = {key.lstrip('0'): value for key, value in json.load(f).items()}
        else:
            self.synsetToCategory = {key.lstrip('0'): value for key, value in self.synsetMapping.items()}

        self.synetToText = dict([(k.lstrip('0'), v) for (k, v) in self.synsetMapping.items()])

        self._loadCsv()

    def _loadCsv(self):
        if not os.path.exists(self.csvFile):
            raise ValueError(f"CSV file {self.csvFile} does not exist.")

        missing = []
        categoryCounts = Counter()
        with open(self.csvFile, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                modelId = row["model_id"]
                synsetId = row["synset_id"].lstrip('0')
                description = row["description"]

                if self.categories and self.synetToText[synsetId] not in self.categories:
                    continue

                textPath = os.path.join(self.rootDir, f"{modelId}_text.npy")
                shapePath = os.path.join(self.rootDir, f"{modelId}_shape.npy")

                if os.path.exists(textPath) and os.path.exists(shapePath):
                    category = self.synsetToCategory.get(synsetId, "Unknown")
                    
                    textEmbedding = np.load(textPath)
                    shapeEmbedding = np.load(shapePath)
                    
                    self.data.append({
                        "category": category,
                        "model_id": modelId,
                        "synset_id": synsetId,
                        "description": description,
                        "text_embedding": textEmbedding,
                        "shape_embedding": shapeEmbedding,
                    })
                    self.data_dict[modelId] = {
                        "category": category,
                        "model_id": modelId,
                        "synset_id": synsetId,
                        "description": description,
                        "text_embedding": textEmbedding,
                        "shape_embedding": shapeEmbedding,
                    }
                    categoryCounts[category] += 1
                else:
                    missing.append(modelId)

        print(f"Category Distribution: {dict(categoryCounts)}")
        if missing:
            print(f"Missing {len(missing)} models from CSV: {missing[:5]}...")

    def split(self, trainRatio=0.8, valRatio=0.1, seed=42):
        random.seed(seed)
        data_indices = list(range(len(self.data)))
        random.shuffle(data_indices)

        trainSize = int(trainRatio * len(self.data))
        valSize = int(valRatio * len(self.data))

        trainIndices = data_indices[:trainSize]
        valIndices = data_indices[trainSize:trainSize + valSize]
        testIndices = data_indices[trainSize + valSize:]

        trainSet = torch.utils.data.Subset(self, trainIndices)
        valSet = torch.utils.data.Subset(self, valIndices)
        testSet = torch.utils.data.Subset(self, testIndices)

        return trainSet, valSet, testSet

    def filterByCategories(self, categories):
        categories = [cat.lstrip('0') for cat in categories]
        filteredData = [item for item in self.data if item["synset_id"] in categories]

        filteredDataset = CLIPEmbeddingDataset(self.rootDir, self.csvFile)
        filteredDataset.categories = categories
        filteredDataset.data = filteredData

        return filteredDataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        textEmbedding = np.load(sample["textPath"])
        shapeEmbedding = np.load(sample["shapePath"])

        return {
            "category": sample["category"],
            "model_id": sample["model_id"],
            "synset_id": sample["synset_id"],
            "description": sample["description"],
            "textEmbedding": torch.from_numpy(textEmbedding).float(),
            "shapeEmbedding": torch.from_numpy(shapeEmbedding).float(),
        }

    def __str__(self):
        """
        Returns a summary of the dataset.
        """
        info = [
            f"CLIPEmbeddingDataset Summary:",
            f"  Root Directory: {self.rootDir}",
            f"  CSV File: {self.csvFile}",
            f"  Total Samples: {len(self.data)}",
            f"  Unique Categories: {len(set(item['category'] for item in self.data))}",
        ]
        if self.categories:
            info.append(f"  Filtered Categories: {', '.join(self.categories)}")
        return "\n".join(info)
