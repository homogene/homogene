"""Tests that the public API is correctly exposed."""

import homogene
from homogene import GenerativeModel, Generator
from homogene.models import GenerativeModel as GenerativeModelFromModels
from homogene.processors import Generator as GeneratorFromProcessors


def test_version():
    assert homogene.__version__ == "0.1.0"


class TestRootPackage:
    def test_generator_importable(self):
        assert Generator is not None

    def test_model_importable(self):
        assert GenerativeModel is not None

    def test_all_contains_generator(self):
        assert "Generator" in homogene.__all__

    def test_all_contains_generative_model(self):
        assert "GenerativeModel" in homogene.__all__


class TestModelsPackage:
    def test_generative_model_importable(self):
        assert GenerativeModelFromModels is not None

    def test_is_same_class(self):
        assert GenerativeModelFromModels is GenerativeModel

    def test_all_contains_generative_model(self):
        import homogene.models
        assert "GenerativeModel" in homogene.models.__all__


class TestProcessorsPackage:
    def test_generator_importable(self):
        assert GeneratorFromProcessors is not None

    def test_is_same_class(self):
        assert GeneratorFromProcessors is Generator

    def test_all_contains_generator(self):
        import homogene.processors
        assert "Generator" in homogene.processors.__all__
