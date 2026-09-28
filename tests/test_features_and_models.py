import numpy as np
import torch

from src.features.handcrafted import extract_features, extract_features_batch, FEATURE_NAMES
from src.models.cnn1d import CNN1DBackbone, CNN1DClassifier, count_params
from src.models.prototypical import PrototypicalNetwork, compute_prototypes, prototypical_logits


def test_extract_features_length_matches_names():
    window = np.random.randn(1024)
    feats = extract_features(window, fs=12000.0)
    assert len(feats) == len(FEATURE_NAMES)
    assert np.all(np.isfinite(feats))


def test_extract_features_batch_shape():
    X = np.random.randn(5, 1024)
    feats = extract_features_batch(X)
    assert feats.shape == (5, len(FEATURE_NAMES))


def test_cnn1d_classifier_output_shape():
    model = CNN1DClassifier(n_classes=10, embedding_dim=32)
    x = torch.randn(4, 1024)
    out = model(x)
    assert out.shape == (4, 10)


def test_cnn1d_under_param_budget():
    model = CNN1DClassifier(n_classes=10, embedding_dim=64)
    assert count_params(model) < 500_000


def test_compute_prototypes_shape():
    embeddings = torch.randn(9, 8)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    protos = compute_prototypes(embeddings, labels, n_classes=3)
    assert protos.shape == (3, 8)


def test_prototypical_logits_shape():
    query = torch.randn(5, 8)
    protos = torch.randn(3, 8)
    logits = prototypical_logits(query, protos)
    assert logits.shape == (5, 3)


def test_protonet_forward_pass():
    model = PrototypicalNetwork(embedding_dim=16)
    support_x = torch.randn(6, 256)
    support_y = torch.tensor([0, 0, 1, 1, 2, 2])
    query_x = torch.randn(4, 256)
    logits = model(support_x, support_y, query_x, n_classes=3)
    assert logits.shape == (4, 3)
