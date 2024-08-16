import numpy as np
import torch
import torch.nn as nn


# Utilities for reading, creating and fetching glove embedding layer.
class GloveUtils:
    def __init__(self, path_to_embeddings, max_dims):
        self.GLOVE_PATH = path_to_embeddings
        self.MAX_DIMS = max_dims
        self.glove_dict = self.read_glove_vectors(self.GLOVE_PATH)
        self.glove_set = None
        self.glove_token_arr = None

    # Read glove embedding from .txt file
    def read_glove_vectors(self, path: str):
        glo_dict = {}
        with open(path, mode='r', encoding='utf8') as file:
            for line in file.readlines():
                line = line.split(" ")
                glo_dict[line[0]] = np.asarray(line[1:], dtype=np.float32)
        return glo_dict

    # Finding common words (not recommended for reproducibility)
    # Using this reduces the memory footprint by finding an intersection of sets of vocabulary,
    # but will generate OOV words if the test data is significantly different from train.
    def update_common_glove_set(self, data: list, glo_dict: dict):
        vant_vocab_set = set()

        for tokens in data:
            vant_vocab_set.update(tokens)

        glove_set = set(glo_dict.keys())
        glove_set &= vant_vocab_set
        print(f"Common vocabulary size: {len(glove_set)}")
        self.glove_set = glove_set

    # Initializes a glove embeddings layer using PyTorch
    def create_glove_emb_layer(self, trainable=False):
        if self.glove_set is None:
            self.glove_set = set(self.glove_dict.keys())
        self.glove_token_arr = np.array(sorted(list(self.glove_set)))

        # single numpy array before building a torch Tensor
        glove_vec_arr = np.asarray([self.glove_dict[token] for token in self.glove_token_arr], dtype=np.float32)
        glove_emb_layer = nn.Embedding.from_pretrained(torch.Tensor(glove_vec_arr))
        glove_emb_layer.weight.requires_grad = trainable
        # only relevant glove embeddings are in memory.
        print(f"Glove Embedding shape: {glove_vec_arr.shape}")
        return glove_emb_layer

    # Returns an index array containing indices of all word vectors in a document
    def __doc2ind(self, doc) -> np.ndarray:
        if self.glove_token_arr is None:
            raise ValueError("Please create an embedding layer first!")

        token_idx = np.minimum(np.searchsorted(self.glove_token_arr, doc), len(self.glove_token_arr) - 1)
        valid_tokens = self.glove_token_arr[token_idx] == doc
        # pre-padding the vector then filling it with valid indices
        indices = np.zeros(self.MAX_DIMS, dtype=np.int32)
        indices[:sum(valid_tokens)] = token_idx[valid_tokens]
        return indices

    # helper function to collect all embedding vectors
    def get_embedding_indices(self, data) -> torch.LongTensor:
        return torch.LongTensor(np.asarray([self.__doc2ind(doc) for doc in data]))
