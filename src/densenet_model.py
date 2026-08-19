from typing import Literal

from torch import nn
from torchvision.models import (
    DenseNet121_Weights,
    densenet121,
)


TrainingMode = Literal[
    "scratch",
    "frozen",
    "last_block",
    "full",
]

TRAINING_MODES = {
    "scratch",
    "frozen",
    "last_block",
    "full",
}


class DenseNet121Classifier(nn.Module):
    feature_dimension = 1024

    def __init__(
        self,
        training_mode: TrainingMode,
        dropout_probability=0.20,
    ):
        super().__init__()

        if training_mode not in TRAINING_MODES:
            allowed = ", ".join(sorted(TRAINING_MODES))
            raise ValueError(
                f"Unknown training mode {training_mode}. "
                f"Expected one of: {allowed}"
            )

        if not 0 <= dropout_probability < 1:
            raise ValueError(
                "dropout_probability must be in [0, 1)"
            )

        self.training_mode = training_mode

        weights = (
            None
            if training_mode == "scratch"
            else DenseNet121_Weights.DEFAULT
        )

        self.backbone = densenet121(weights=weights)

        backbone_feature_dimension = (
            self.backbone.classifier.in_features
        )

        if (
            backbone_feature_dimension
            != self.feature_dimension
        ):
            raise ValueError(
                "Unexpected DenseNet121 feature dimension: "
                f"{backbone_feature_dimension}"
            )

        self.backbone.classifier = nn.Identity()

        self.dropout = nn.Dropout(
            p=dropout_probability
        )
        self.classifier = nn.Linear(
            self.feature_dimension,
            2,
        )

        nn.init.normal_(
            self.classifier.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(self.classifier.bias)

        self._configure_trainable_parameters()

    def _configure_trainable_parameters(self):
        for parameter in self.parameters():
            parameter.requires_grad = True

        if self.training_mode == "frozen":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        if self.training_mode == "last_block":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

            for parameter in (
                self.backbone
                .features
                .denseblock4
                .parameters()
            ):
                parameter.requires_grad = True

            for parameter in (
                self.backbone
                .features
                .norm5
                .parameters()
            ):
                parameter.requires_grad = True

    def _set_frozen_modules_to_evaluation(self):
        if self.training_mode == "frozen":
            self.backbone.eval()
            return

        if self.training_mode == "last_block":
            trainable_feature_modules = {
                "denseblock4",
                "norm5",
            }

            for name, module in (
                self.backbone.features.named_children()
            ):
                if name not in trainable_feature_modules:
                    module.eval()

    def train(self, mode=True):
        super().train(mode)

        if mode:
            self._set_frozen_modules_to_evaluation()

        return self

    def forward_features(self, images):
        return self.backbone(images)

    def forward(self, images):
        features = self.forward_features(images)
        return self.classifier(
            self.dropout(features)
        )


def count_parameters(model):
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
    }