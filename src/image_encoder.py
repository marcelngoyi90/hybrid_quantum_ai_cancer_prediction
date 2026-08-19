import torch
from torch import nn
from torchvision.models import (
    DenseNet121_Weights,
    densenet121,
)


class ImageProjectionBlock(nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        dropout_probability,
    ):
        super().__init__()

        if input_dimension <= 0:
            raise ValueError(
                "Input dimension must be positive"
            )

        if output_dimension <= 0:
            raise ValueError(
                "Output dimension must be positive"
            )

        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError(
                "Dropout probability must be in "
                "[0, 1)"
            )

        self.linear = nn.Linear(
            input_dimension,
            output_dimension,
        )
        self.normalization = nn.LayerNorm(
            output_dimension
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(
            dropout_probability
        )

    def forward(self, inputs):
        hidden = self.linear(inputs)
        hidden = self.normalization(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        return hidden


class DenseNet121ImagePath(nn.Module):
    def __init__(
        self,
        input_height,
        input_width,
        input_channels,
        encoder_output_dimension,
        projector_hidden_dimension,
        projector_output_dimension,
        dropout_probability,
    ):
        super().__init__()

        self.input_height = int(
            input_height
        )
        self.input_width = int(
            input_width
        )
        self.input_channels = int(
            input_channels
        )
        self.encoder_output_dimension = int(
            encoder_output_dimension
        )
        self.projector_hidden_dimension = int(
            projector_hidden_dimension
        )
        self.projector_output_dimension = int(
            projector_output_dimension
        )

        if self.input_height <= 0:
            raise ValueError(
                "Input height must be positive"
            )

        if self.input_width <= 0:
            raise ValueError(
                "Input width must be positive"
            )

        if self.input_channels != 3:
            raise ValueError(
                "DenseNet121 requires three-channel "
                "RGB input"
            )

        if self.encoder_output_dimension != 1024:
            raise ValueError(
                "DenseNet121 encoder output must be "
                "1,024 dimensions"
            )

        self.backbone = densenet121(
            weights=(
                DenseNet121_Weights.DEFAULT
            )
        )

        detected_feature_dimension = int(
            self.backbone.classifier.in_features
        )

        if detected_feature_dimension != (
            self.encoder_output_dimension
        ):
            raise ValueError(
                "Unexpected DenseNet121 feature "
                f"dimension: "
                f"{detected_feature_dimension}"
            )

        self.backbone.classifier = nn.Identity()

        for parameter in (
            self.backbone.parameters()
        ):
            parameter.requires_grad = True

        self.projector = nn.Sequential(
            ImageProjectionBlock(
                input_dimension=(
                    self.encoder_output_dimension
                ),
                output_dimension=(
                    self.projector_hidden_dimension
                ),
                dropout_probability=(
                    dropout_probability
                ),
            ),
            ImageProjectionBlock(
                input_dimension=(
                    self.projector_hidden_dimension
                ),
                output_dimension=(
                    self.projector_output_dimension
                ),
                dropout_probability=(
                    dropout_probability
                ),
            ),
        )

    def _validate_images(
        self,
        images,
    ):
        if images.ndim != 4:
            raise ValueError(
                "Images must be shaped "
                "(batch, channels, height, width)"
            )

        if images.shape[1] != (
            self.input_channels
        ):
            raise ValueError(
                "Unexpected image channel count"
            )

        if images.shape[2] != (
            self.input_height
        ):
            raise ValueError(
                "Unexpected image height"
            )

        if images.shape[3] != (
            self.input_width
        ):
            raise ValueError(
                "Unexpected image width"
            )

        if not images.is_floating_point():
            raise TypeError(
                "Images must use a floating point "
                "dtype"
            )

    def forward_features(
        self,
        images,
    ):
        self._validate_images(
            images
        )

        features = self.backbone(
            images
        )

        if features.ndim != 2:
            raise RuntimeError(
                "DenseNet121 returned features with "
                "an unexpected rank"
            )

        if features.shape[1] != (
            self.encoder_output_dimension
        ):
            raise RuntimeError(
                "DenseNet121 returned an unexpected "
                "feature dimension"
            )

        return features

    def project(
        self,
        encoded_features,
    ):
        if encoded_features.ndim != 2:
            raise ValueError(
                "Encoded image features must be "
                "two-dimensional"
            )

        if encoded_features.shape[1] != (
            self.encoder_output_dimension
        ):
            raise ValueError(
                "Unexpected encoded image dimension"
            )

        projected = self.projector(
            encoded_features
        )

        if projected.shape[1] != (
            self.projector_output_dimension
        ):
            raise RuntimeError(
                "Image projector returned an "
                "unexpected dimension"
            )

        return projected

    def forward(
        self,
        images,
    ):
        encoded = self.forward_features(
            images
        )

        return self.project(
            encoded
        )


def build_densenet121_image_path(
    joint_config,
):
    image_config = joint_config[
        "image_path"
    ]

    if image_config[
        "architecture"
    ] != "densenet121":
        raise ValueError(
            "Only DenseNet121 is supported"
        )

    if image_config[
        "initialization"
    ] != "imagenet":
        raise ValueError(
            "The joint image path requires "
            "ImageNet initialization"
        )

    if image_config[
        "fine_tuning"
    ] != "all_layers":
        raise ValueError(
            "The joint image path requires full "
            "DenseNet121 fine-tuning"
        )

    if image_config[
        "normalization"
    ] != "imagenet":
        raise ValueError(
            "DenseNet121 requires ImageNet "
            "normalization"
        )

    return DenseNet121ImagePath(
        input_height=int(
            image_config[
                "input_height"
            ]
        ),
        input_width=int(
            image_config[
                "input_width"
            ]
        ),
        input_channels=int(
            image_config[
                "input_channels"
            ]
        ),
        encoder_output_dimension=int(
            image_config[
                "encoder_output_dimension"
            ]
        ),
        projector_hidden_dimension=int(
            image_config[
                "projector_hidden_dimension"
            ]
        ),
        projector_output_dimension=int(
            image_config[
                "projector_output_dimension"
            ]
        ),
        dropout_probability=float(
            image_config[
                "dropout_probability"
            ]
        ),
    )
