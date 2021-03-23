"""
GatedConv as in gatedconv.py (where bond is represented as node), but bond as edge graph.
"""

from typing import Callable, Dict, Union

import dgl
import torch
from dgl import function as fn
from torch import nn

from rxnrep.model.utils import MLP


class GatedGCNConv(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_fc_layers: int = 1,
        batch_norm: bool = True,
        activation: Callable = nn.ReLU(),
        residual: bool = False,
        dropout: Union[float, None] = None,
    ):
        super(GatedGCNConv, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.residual = residual

        if in_size != out_size:
            self.residual = False

        # A, B, ... I are phi_1, phi_2, ..., phi_9 in the BonDNet paper
        hidden = [out_size] * (num_fc_layers - 1)
        self.A = MLP(in_size, hidden, activation=activation, out_size=out_size)
        self.B = MLP(in_size, hidden, activation=activation, out_size=out_size)
        self.C = MLP(in_size, hidden, activation=activation, out_size=out_size)
        self.D = MLP(in_size, hidden, activation=activation, out_size=out_size)
        self.E = MLP(in_size, hidden, activation=activation, out_size=out_size)
        self.F = MLP(in_size, hidden, activation=activation, out_size=out_size)
        self.G = MLP(out_size, hidden, activation=activation, out_size=out_size)
        self.H = MLP(out_size, hidden, activation=activation, out_size=out_size)
        self.I = MLP(in_size, hidden, activation=activation, out_size=out_size)

        if self.batch_norm:
            self.bn_node_h = nn.BatchNorm1d(out_size)
            self.bn_node_e = nn.BatchNorm1d(out_size)
            self.bn_node_u = nn.BatchNorm1d(out_size)

        delta = 1e-3
        if dropout is None or dropout < delta:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        g: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            g: the graphm bond as edges
            feats: {name, feats}. Atom, bond, and global features.

        Returns:
            updated features.
        """

        g = g.local_var()

        h = feats["atom"]
        e = feats["bond"]
        u = feats["global"]

        #
        # update bond feature e
        #
        g.nodes["atom"].data.update({"Ah": self.A(h)})
        g.edges["bond"].data.update({"Be": self.B(e)})
        g.nodes["global"].data.update({"Cu": self.C(u)})

        # step 1
        # global feats to atom node (a simpy copy, sum operates on 1 tensor,
        # since we have only one global nodes connected to each atom node)
        g.update_all(fn.copy_u("Cu", "m"), fn.sum("m", "Cu"), etype="g2a")

        # step 2, aggregate edge features
        # sum of:
        # src and dst atom feats,
        # edge feats,
        # and global feats (already stored in atom nodes, here we only grab it from src)
        g.apply_edges(
            lambda edges: {
                "e": edges.src["Ah"]
                + edges.dst["Ah"]
                + edges.data["Be"]
                + edges.src["Cu"],
            },
            etype="bond",
        )
        e = g.edges["bond"].data["e"]

        # del for memory efficiency
        del g.nodes["atom"].data["Ah"]
        del g.edges["bond"].data["Be"]
        del g.nodes["global"].data["Cu"]

        if self.batch_norm:
            e = self.bn_node_e(e)
        e = self.activation(e)
        if self.residual:
            e = feats["bond"] + e

        #
        # update atom feature h
        #

        # step 1
        # edge feats to atom nodes: sum_j e_ij [Had] Eh_j
        g.nodes["atom"].data.update({"Eh": self.E(h)})
        g.edges["bond"].data["e"] = e
        g.update_all(atom_message_fn, atom_reduce_fn, etype="bond")
        try:
            h1 = g.nodes["atom"].data.pop("h1")
        except KeyError:
            # This only happens when there is no edges (e.g. single atom molecule H+).
            # Will not happen when the single atom molecule is batched with other
            # molecules. When batched, the batched graph has edges; thus `atom_reduce_fn`
            # will be called, and `h1` for the single atom molecule is initialized to a
            # zero tensor by dgl.
            h1 = 0.0

        # step 2
        # global feats to atom node (a simpy copy, sum operates on 1 tensor,
        # since we have only one global nodes connected to each atom node)
        g.nodes["global"].data.update({"Fu": self.F(u)})
        g.update_all(fn.copy_u("Fu", "m"), fn.sum("m", "Fu"), etype="g2a")
        h2 = g.nodes["atom"].data.pop("Fu")

        h = self.D(h) + h1 + h2

        # del for memory efficiency
        del g.nodes["atom"].data["Eh"]

        if self.batch_norm:
            h = self.bn_node_h(h)
        h = self.activation(h)
        if self.residual:
            h = feats["atom"] + h

        #
        # update global feature u
        #
        g.nodes["atom"].data["Gh"] = self.G(h)
        g.edges["bond"].data["He"] = self.H(e)

        # edge feats to global
        # Each bond has two edges, we do not need to divide 2 since divide by num_edges
        # already takes care of it
        g.update_all(fn.copy_e("He", "m"), fn.sum("m", "He_sum"), etype="bond")
        g.update_all(fn.copy_u("He_sum", "m"), fn.sum("m", "He_sum"), etype="a2g")
        num_edges = g.num_edges("bond")
        if num_edges == 0:
            # single atom molecule
            num_edges = 1
        mean_He = g.nodes["global"].data.pop("He_sum") / num_edges

        # atom nodes to global
        g.update_all(fn.copy_u("Gh", "m"), fn.mean("m", "Gh_sum"), etype="a2g")
        mean_Gh = g.nodes["global"].data.pop("Gh_sum")

        # aggregate
        u = mean_Gh + mean_He + self.I(u)

        if self.batch_norm:
            # do not apply batch norm if it there is only one graph and it is in
            # training mode, BN complains about it
            if u.shape[0] <= 1 and self.training:
                pass
            else:
                u = self.bn_node_u(u)
        u = self.activation(u)
        if self.residual:
            u = feats["global"] + u

        # dropout
        h = self.dropout(h)
        e = self.dropout(e)
        u = self.dropout(u)

        feats = {"atom": h, "bond": e, "global": u}

        return feats


def atom_message_fn(edges):
    return {
        "Eh_j": edges.src["Eh"],
        "sigma_eij": torch.sigmoid(edges.data["e"]),
    }


def atom_reduce_fn(nodes):
    Eh_j = nodes.mailbox["Eh_j"]
    sigma_eij = nodes.mailbox["sigma_eij"]

    # (sum_j eta_ij * Ehj)/(sum_j' eta_ij') <= dense attention
    h1 = torch.sum(sigma_eij * Eh_j, dim=1) / (torch.sum(sigma_eij, dim=1) + 1e-6)

    return {"h1": h1}
