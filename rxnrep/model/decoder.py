import torch
import torch.nn as nn
from rxnrep.model.utils import FCNN
from typing import List


class BaseDecoder(nn.Module):
    """
    A base decoder class to map the features before passing them to sigmoid or softmax
    layer to compute logits.

    This uses N fully connected layer as the decoder.

    Args:
        in_size: input size of the features
        num_classes: number of classes
        hidden_layer_sizes: size of the hidden layers to transform the features.
            Note, there will be an additional layer applied after this,
            which transforms the features to `num_classes` dimensions.
        activation: activation function applied after the hidden layer
    """

    def __init__(
        self,
        in_size: int,
        num_classes: int,
        hidden_layer_sizes: List[int] = None,
        activation: str = "ReLU",
    ):
        super(BaseDecoder, self).__init__()

        # set default values
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64]
        if isinstance(activation, str):
            activation = getattr(nn, activation)()

        # no activation for last layer
        out_sizes = hidden_layer_sizes + [num_classes]
        use_bias = [True] * len(out_sizes)
        acts = [activation] * len(hidden_layer_sizes) + [None]

        self.fc_layers = FCNN(in_size, out_sizes, acts, use_bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: features, a 2D tensor of shape (N, D), where N is the batch size and
                D is the feature dimension.

        Returns:
            Updated features, a 2D tensor of shape (N, num_classes), where `num_classes`
            is the number of classes.
        """

        return self.fc_layers(feats)


class BondTypeDecoder(BaseDecoder):
    """
    A decoder to predict the bond type from bond features.

    There are three types of bond:
    1. unchanged bond: bonds exists in both the reactants and the products
    2. lost bond: bonds in the reactants breaks in a reaction
    3. added bonds: bonds in the products created in a reaction

    Args:
        in_size: input size of the features
        num_classes: number of classes
        hidden_layer_sizes: size of the hidden layers to transform the features.
            Note, there will be an additional layer applied after this,
            which transforms the features to `num_classes` dimensions.
        activation: activation function applied after the hidden layer
    """

    def __init__(
        self,
        in_size: int,
        num_classes: int = 3,
        hidden_layer_sizes: List[int] = None,
        activation: str = "ReLU",
    ):
        super(BondTypeDecoder, self).__init__(
            in_size, num_classes, hidden_layer_sizes, activation
        )


class AtomInReactionCenterDecoder(BaseDecoder):
    """
    A decoder to predict whether an atom is in the reaction center from the atom features.

    The are two classes:
    0: not in reaction center, 1: in reaction center

    Args:
        in_size: input size of the features
        hidden_layer_sizes: size of the hidden layers to transform the features.
            Note, there will be an additional layer applied after this,
            which transforms the features to `num_classes` dimensions.
        activation: activation function applied after the hidden layer
    """

    def __init__(
        self,
        in_size: int,
        hidden_layer_sizes: List[int] = None,
        activation: str = "ReLU",
    ):
        # Note, for binary classification, the sigmoid function takes a scalar,
        # so `num_classes` is set to 1
        num_classes = 1
        super(AtomInReactionCenterDecoder, self).__init__(
            in_size, num_classes, hidden_layer_sizes, activation,
        )


class ReactionClusterDecoder(BaseDecoder):
    """
    A decoder to predict the clustering labels from reaction features.

    Args:
        in_size: input size of the features
        num_classes: number of classes
        hidden_layer_sizes: size of the hidden layers to transform the features.
            Note, there will be an additional layer applied after this,
            which transforms the features to `num_classes` dimensions.
        activation: activation function applied after the hidden layer
    """

    def __init__(
        self,
        in_size: int,
        num_classes: int,
        hidden_layer_sizes: List[int] = None,
        activation: str = "ReLU",
    ):
        super(ReactionClusterDecoder, self).__init__(
            in_size, num_classes, hidden_layer_sizes, activation,
        )
