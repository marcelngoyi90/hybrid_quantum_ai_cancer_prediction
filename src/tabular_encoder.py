import torch
from torch import nn


class FeedForwardBlock(nn.Module):
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


class SeerTabularPath(nn.Module):
    def __init__(
        self,
        numeric_features,
        categorical_features,
        num_embeddings,
        embedding_dimensions,
        encoder_hidden_dimensions,
        encoder_output_dimension,
        projector_output_dimension,
        dropout_probability,
        unknown_category_index=0,
    ):
        super().__init__()

        self.numeric_features = list(
            numeric_features
        )
        self.categorical_features = list(
            categorical_features
        )
        self.unknown_category_index = int(
            unknown_category_index
        )

        if not self.numeric_features:
            raise ValueError(
                "At least one numeric feature is "
                "required"
            )

        if not self.categorical_features:
            raise ValueError(
                "At least one categorical feature is "
                "required"
            )

        if self.unknown_category_index != 0:
            raise ValueError(
                "Unknown category index must be 0"
            )

        if set(num_embeddings) != set(
            self.categorical_features
        ):
            raise ValueError(
                "Embedding-table sizes do not match "
                "the categorical feature list"
            )

        if set(embedding_dimensions) != set(
            self.categorical_features
        ):
            raise ValueError(
                "Embedding dimensions do not match "
                "the categorical feature list"
            )

        self.num_embeddings_by_feature = {
            feature: int(
                num_embeddings[feature]
            )
            for feature
            in self.categorical_features
        }

        self.embedding_dimensions_by_feature = {
            feature: int(
                embedding_dimensions[feature]
            )
            for feature
            in self.categorical_features
        }

        for feature in (
            self.categorical_features
        ):
            if (
                self.num_embeddings_by_feature[
                    feature
                ]
                <= self.unknown_category_index
            ):
                raise ValueError(
                    "Invalid embedding-table size for "
                    f"{feature}"
                )

            if (
                self.embedding_dimensions_by_feature[
                    feature
                ]
                <= 0
            ):
                raise ValueError(
                    "Invalid embedding dimension for "
                    f"{feature}"
                )

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=(
                        self
                        .num_embeddings_by_feature[
                            feature
                        ]
                    ),
                    embedding_dim=(
                        self
                        .embedding_dimensions_by_feature[
                            feature
                        ]
                    ),
                )
                for feature
                in self.categorical_features
            ]
        )

        self._initialize_unknown_embeddings()

        self.numeric_input_dimension = len(
            self.numeric_features
        )

        self.embedding_output_dimension = sum(
            self.embedding_dimensions_by_feature[
                feature
            ]
            for feature
            in self.categorical_features
        )

        self.combined_input_dimension = (
            self.numeric_input_dimension
            + self.embedding_output_dimension
        )

        hidden_dimensions = [
            int(dimension)
            for dimension
            in encoder_hidden_dimensions
        ]

        if not hidden_dimensions:
            raise ValueError(
                "Encoder hidden dimensions cannot "
                "be empty"
            )

        if any(
            dimension <= 0
            for dimension in hidden_dimensions
        ):
            raise ValueError(
                "Encoder hidden dimensions must be "
                "positive"
            )

        if hidden_dimensions[-1] != int(
            encoder_output_dimension
        ):
            raise ValueError(
                "The last encoder hidden dimension "
                "must equal encoder_output_dimension"
            )

        encoder_dimensions = [
            self.combined_input_dimension,
            *hidden_dimensions,
        ]

        self.encoder_blocks = nn.ModuleList(
            [
                FeedForwardBlock(
                    input_dimension=(
                        encoder_dimensions[index]
                    ),
                    output_dimension=(
                        encoder_dimensions[index + 1]
                    ),
                    dropout_probability=(
                        dropout_probability
                    ),
                )
                for index in range(
                    len(encoder_dimensions) - 1
                )
            ]
        )

        self.encoder_output_dimension = int(
            encoder_output_dimension
        )
        self.projector_output_dimension = int(
            projector_output_dimension
        )

        self.projector = FeedForwardBlock(
            input_dimension=(
                self.encoder_output_dimension
            ),
            output_dimension=(
                self.projector_output_dimension
            ),
            dropout_probability=(
                dropout_probability
            ),
        )

    def _initialize_unknown_embeddings(self):
        with torch.no_grad():
            for embedding in self.embeddings:
                embedding.weight[
                    self.unknown_category_index
                ].zero_()

    def _validate_inputs(
        self,
        numeric_inputs,
        categorical_inputs,
    ):
        if numeric_inputs.ndim != 2:
            raise ValueError(
                "Numeric inputs must be shaped "
                "(batch, numeric_features)"
            )

        if categorical_inputs.ndim != 2:
            raise ValueError(
                "Categorical inputs must be shaped "
                "(batch, categorical_features)"
            )

        if (
            numeric_inputs.shape[0]
            != categorical_inputs.shape[0]
        ):
            raise ValueError(
                "Numeric and categorical batch sizes "
                "must match"
            )

        if numeric_inputs.shape[1] != len(
            self.numeric_features
        ):
            raise ValueError(
                "Unexpected numeric feature count"
            )

        if categorical_inputs.shape[1] != len(
            self.categorical_features
        ):
            raise ValueError(
                "Unexpected categorical feature count"
            )

        if not numeric_inputs.is_floating_point():
            raise TypeError(
                "Numeric inputs must use a floating "
                "point dtype"
            )

        if categorical_inputs.dtype != torch.int64:
            raise TypeError(
                "Categorical inputs must use "
                "torch.int64"
            )

    def embed_categorical(
        self,
        categorical_inputs,
    ):
        embedded_features = []

        for column_index, embedding in enumerate(
            self.embeddings
        ):
            feature_indices = categorical_inputs[
                :,
                column_index,
            ]

            embedded_features.append(
                embedding(feature_indices)
            )

        return torch.cat(
            embedded_features,
            dim=1,
        )

    def encode(
        self,
        numeric_inputs,
        categorical_inputs,
    ):
        self._validate_inputs(
            numeric_inputs,
            categorical_inputs,
        )

        embedded = self.embed_categorical(
            categorical_inputs
        )

        hidden = torch.cat(
            [
                numeric_inputs,
                embedded,
            ],
            dim=1,
        )

        for block in self.encoder_blocks:
            hidden = block(hidden)

        return hidden

    def project(
        self,
        encoded_features,
    ):
        if encoded_features.ndim != 2:
            raise ValueError(
                "Encoded features must be "
                "two-dimensional"
            )

        if encoded_features.shape[1] != (
            self.encoder_output_dimension
        ):
            raise ValueError(
                "Unexpected encoded feature dimension"
            )

        return self.projector(
            encoded_features
        )

    def forward(
        self,
        numeric_inputs,
        categorical_inputs,
    ):
        encoded = self.encode(
            numeric_inputs,
            categorical_inputs,
        )

        return self.project(
            encoded
        )


def build_seer_tabular_path(
    joint_config,
    preprocessor_state,
):
    tabular_config = joint_config[
        "tabular_path"
    ]

    if tabular_config[
        "activation"
    ] != "gelu":
        raise ValueError(
            "Only GELU activation is supported"
        )

    if tabular_config[
        "normalization"
    ] != "layer_norm":
        raise ValueError(
            "Only layer normalization is supported"
        )

    numeric_features = tabular_config[
        "numeric_features"
    ]
    categorical_features = tabular_config[
        "categorical_features"
    ]

    if preprocessor_state[
        "numeric_features"
    ] != numeric_features:
        raise ValueError(
            "Preprocessor numeric feature order "
            "does not match the model"
        )

    if preprocessor_state[
        "categorical_features"
    ] != categorical_features:
        raise ValueError(
            "Preprocessor categorical feature order "
            "does not match the model"
        )

    num_embeddings = {
        feature: int(
            preprocessor_state[
                "num_embeddings"
            ][feature]
        )
        for feature
        in categorical_features
    }

    embedding_dimensions = {
        feature: int(
            preprocessor_state[
                "embedding_dimensions"
            ][feature]
        )
        for feature
        in categorical_features
    }

    expected_cardinalities = tabular_config[
        "training_category_cardinalities"
    ]

    for feature in categorical_features:
        expected_num_embeddings = (
            int(
                expected_cardinalities[
                    feature
                ]
            )
            + 1
        )

        if num_embeddings[
            feature
        ] != expected_num_embeddings:
            raise ValueError(
                "Embedding-table size changed for "
                f"{feature}"
            )

    return SeerTabularPath(
        numeric_features=numeric_features,
        categorical_features=(
            categorical_features
        ),
        num_embeddings=num_embeddings,
        embedding_dimensions=(
            embedding_dimensions
        ),
        encoder_hidden_dimensions=(
            tabular_config[
                "encoder_hidden_dimensions"
            ]
        ),
        encoder_output_dimension=(
            tabular_config[
                "encoder_output_dimension"
            ]
        ),
        projector_output_dimension=(
            tabular_config[
                "projector_output_dimension"
            ]
        ),
        dropout_probability=float(
            tabular_config[
                "dropout_probability"
            ]
        ),
        unknown_category_index=int(
            tabular_config[
                "categorical_preprocessing"
            ][
                "unknown_category_index"
            ]
        ),
    )
