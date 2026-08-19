from torch import nn

from src.image_encoder import (
    build_densenet121_image_path,
)
from src.tabular_encoder import (
    FeedForwardBlock,
    build_seer_tabular_path,
)


class SharedTrunk(nn.Module):
    def __init__(
        self,
        input_dimension,
        hidden_dimension,
        output_dimension,
        dropout_probability,
    ):
        super().__init__()

        self.input_dimension = int(
            input_dimension
        )
        self.hidden_dimension = int(
            hidden_dimension
        )
        self.output_dimension = int(
            output_dimension
        )

        if self.input_dimension <= 0:
            raise ValueError(
                "Shared input dimension must be "
                "positive"
            )

        if self.hidden_dimension <= 0:
            raise ValueError(
                "Shared hidden dimension must be "
                "positive"
            )

        if self.output_dimension <= 0:
            raise ValueError(
                "Shared output dimension must be "
                "positive"
            )

        self.blocks = nn.Sequential(
            FeedForwardBlock(
                input_dimension=(
                    self.input_dimension
                ),
                output_dimension=(
                    self.hidden_dimension
                ),
                dropout_probability=(
                    dropout_probability
                ),
            ),
            FeedForwardBlock(
                input_dimension=(
                    self.hidden_dimension
                ),
                output_dimension=(
                    self.output_dimension
                ),
                dropout_probability=(
                    dropout_probability
                ),
            ),
        )

    def forward(
        self,
        projected_features,
    ):
        if projected_features.ndim != 2:
            raise ValueError(
                "Projected features must be "
                "two-dimensional"
            )

        if projected_features.shape[1] != (
            self.input_dimension
        ):
            raise ValueError(
                "Unexpected shared-trunk input "
                "dimension"
            )

        shared_features = self.blocks(
            projected_features
        )

        if shared_features.shape[1] != (
            self.output_dimension
        ):
            raise RuntimeError(
                "Shared trunk returned an unexpected "
                "dimension"
            )

        return shared_features


class BinaryTaskHead(nn.Module):
    def __init__(
        self,
        input_dimension,
    ):
        super().__init__()

        if input_dimension <= 0:
            raise ValueError(
                "Task-head input dimension must be "
                "positive"
            )

        self.input_dimension = int(
            input_dimension
        )

        self.classifier = nn.Linear(
            self.input_dimension,
            1,
        )

        nn.init.normal_(
            self.classifier.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(
            self.classifier.bias
        )

    def forward(
        self,
        shared_features,
    ):
        if shared_features.ndim != 2:
            raise ValueError(
                "Shared features must be "
                "two-dimensional"
            )

        if shared_features.shape[1] != (
            self.input_dimension
        ):
            raise ValueError(
                "Unexpected task-head input "
                "dimension"
            )

        logits = self.classifier(
            shared_features
        )

        if (
            logits.ndim != 2
            or logits.shape[1] != 1
        ):
            raise RuntimeError(
                "Binary task head returned an "
                "unexpected shape"
            )

        return logits.squeeze(
            dim=1
        )


class JointCancerModel(nn.Module):
    def __init__(
        self,
        image_path,
        tabular_path,
        shared_trunk,
        image_head,
        tabular_head,
    ):
        super().__init__()

        self.image_path = image_path
        self.tabular_path = tabular_path
        self.shared_trunk = shared_trunk
        self.image_head = image_head
        self.tabular_head = tabular_head

        image_projection_dimension = (
            self.image_path
            .projector_output_dimension
        )
        tabular_projection_dimension = (
            self.tabular_path
            .projector_output_dimension
        )

        if image_projection_dimension != (
            self.shared_trunk.input_dimension
        ):
            raise ValueError(
                "Image projection dimension does not "
                "match the shared trunk"
            )

        if tabular_projection_dimension != (
            self.shared_trunk.input_dimension
        ):
            raise ValueError(
                "Tabular projection dimension does "
                "not match the shared trunk"
            )

        if self.image_head.input_dimension != (
            self.shared_trunk.output_dimension
        ):
            raise ValueError(
                "Image-head dimension does not match "
                "the shared trunk"
            )

        if self.tabular_head.input_dimension != (
            self.shared_trunk.output_dimension
        ):
            raise ValueError(
                "Tabular-head dimension does not "
                "match the shared trunk"
            )

    def encode_image(
        self,
        images,
    ):
        projected = self.image_path(
            images
        )
        shared = self.shared_trunk(
            projected
        )

        return projected, shared

    def encode_tabular(
        self,
        numeric_inputs,
        categorical_inputs,
    ):
        projected = self.tabular_path(
            numeric_inputs,
            categorical_inputs,
        )
        shared = self.shared_trunk(
            projected
        )

        return projected, shared

    def forward_image(
        self,
        images,
    ):
        _, shared = self.encode_image(
            images
        )

        return self.image_head(
            shared
        )

    def forward_tabular(
        self,
        numeric_inputs,
        categorical_inputs,
    ):
        _, shared = self.encode_tabular(
            numeric_inputs,
            categorical_inputs,
        )

        return self.tabular_head(
            shared
        )

    def shared_parameters(
        self,
    ):
        return list(
            self.shared_trunk.parameters()
        )

    def optimizer_parameter_groups(
        self,
    ):
        groups = {
            "image_encoder": list(
                self.image_path
                .backbone
                .parameters()
            ),
            "image_projector": list(
                self.image_path
                .projector
                .parameters()
            ),
            "tabular_encoder": (
                list(
                    self.tabular_path
                    .embeddings
                    .parameters()
                )
                + list(
                    self.tabular_path
                    .encoder_blocks
                    .parameters()
                )
            ),
            "tabular_projector": list(
                self.tabular_path
                .projector
                .parameters()
            ),
            "shared_layers": (
                self.shared_parameters()
            ),
            "task_heads": (
                list(
                    self.image_head.parameters()
                )
                + list(
                    self.tabular_head.parameters()
                )
            ),
        }

        grouped_parameters = [
            parameter
            for parameters in groups.values()
            for parameter in parameters
        ]

        grouped_parameter_ids = [
            id(parameter)
            for parameter
            in grouped_parameters
        ]

        if len(grouped_parameter_ids) != len(
            set(grouped_parameter_ids)
        ):
            raise RuntimeError(
                "A trainable parameter appears in "
                "more than one optimizer group"
            )

        trainable_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

        if set(grouped_parameter_ids) != {
            id(parameter)
            for parameter
            in trainable_parameters
        }:
            raise RuntimeError(
                "Optimizer groups do not cover every "
                "trainable parameter exactly once"
            )

        if any(
            not parameters
            for parameters in groups.values()
        ):
            raise RuntimeError(
                "An optimizer parameter group is empty"
            )

        return groups


def build_joint_cancer_model(
    joint_config,
    preprocessor_state,
):
    shared_config = joint_config[
        "shared_path"
    ]
    head_config = joint_config[
        "task_heads"
    ]

    if shared_config[
        "activation"
    ] != "gelu":
        raise ValueError(
            "Only GELU shared activation is "
            "supported"
        )

    if shared_config[
        "normalization"
    ] != "layer_norm":
        raise ValueError(
            "Only shared layer normalization is "
            "supported"
        )

    image_projection_dimension = int(
        joint_config[
            "image_path"
        ][
            "projector_output_dimension"
        ]
    )

    tabular_projection_dimension = int(
        joint_config[
            "tabular_path"
        ][
            "projector_output_dimension"
        ]
    )

    shared_input_dimension = int(
        shared_config[
            "input_dimension"
        ]
    )

    if (
        image_projection_dimension
        != shared_input_dimension
    ):
        raise ValueError(
            "Image path does not produce the shared "
            "input dimension"
        )

    if (
        tabular_projection_dimension
        != shared_input_dimension
    ):
        raise ValueError(
            "Tabular path does not produce the shared "
            "input dimension"
        )

    shared_output_dimension = int(
        shared_config[
            "output_dimension"
        ]
    )

    for task_name in (
        "image",
        "tabular",
    ):
        task_head_config = (
            head_config[
                task_name
            ]
        )

        if int(
            task_head_config[
                "input_dimension"
            ]
        ) != shared_output_dimension:
            raise ValueError(
                f"{task_name} head input dimension "
                "does not match the shared output"
            )

        if int(
            task_head_config[
                "output_dimension"
            ]
        ) != 1:
            raise ValueError(
                f"{task_name} head must produce one "
                "binary logit"
            )

    image_path = (
        build_densenet121_image_path(
            joint_config
        )
    )

    tabular_path = (
        build_seer_tabular_path(
            joint_config,
            preprocessor_state,
        )
    )

    shared_trunk = SharedTrunk(
        input_dimension=(
            shared_input_dimension
        ),
        hidden_dimension=int(
            shared_config[
                "hidden_dimension"
            ]
        ),
        output_dimension=(
            shared_output_dimension
        ),
        dropout_probability=float(
            shared_config[
                "dropout_probability"
            ]
        ),
    )

    image_head = BinaryTaskHead(
        input_dimension=(
            shared_output_dimension
        )
    )

    tabular_head = BinaryTaskHead(
        input_dimension=(
            shared_output_dimension
        )
    )

    model = JointCancerModel(
        image_path=image_path,
        tabular_path=tabular_path,
        shared_trunk=shared_trunk,
        image_head=image_head,
        tabular_head=tabular_head,
    )

    model.optimizer_parameter_groups()

    return model
