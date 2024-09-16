import numpy as np
import random
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
import os
import my_Features as features
from utils import load_GO_annot_prog

# Load protein list, annotations, GO terms, GO names, and counts from the annotation file
prot_list, prot2annot, goterms, gonames, counts = load_GO_annot_prog(
    "../data/nrPDB-GO_annot.tsv"
)


def _normalize(tensor, dim=-1):
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )  # Normalize by dividing by the norm along the specified dimension, replace NaNs with zeros


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, shape [...dims],if `D` has  then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = torch.linspace(
        D_min, D_max, D_count, device=device
    )  # Create evenly spaced RBF centers between D_min and D_max
    D_mu = D_mu.view([1, -1])  # Reshape to a row vector
    D_sigma = (D_max - D_min) / D_count  # Calculate the standard deviation for RBF
    D_expand = torch.unsqueeze(
        D, -1
    )  # Expand D to have an extra dimension for RBF computation

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))  # Compute the RBF embeddings
    return RBF


class BatchSampler(data.Sampler):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design.

    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.

    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    """

    def __init__(self, node_counts, max_nodes=3000, shuffle=True):
        # Initialize the batch sampler with node counts, max nodes per batch, and shuffle option
        self.node_counts = node_counts  # Store node counts
        self.idx = [
            i for i in range(len(node_counts)) if node_counts[i] <= max_nodes
        ]  # Filter indices where node count is within the allowed maximum
        self.shuffle = shuffle  # Store shuffle option
        self.max_nodes = max_nodes  # Store maximum nodes per batch
        self._form_batches()  # Form initial batches

    def _form_batches(self):
        # Form batches based on node counts and max nodes per batch
        self.batches = []  # Initialize empty list for batches
        if self.shuffle:
            random.shuffle(self.idx)  # Shuffle indices if shuffle option is enabled
        idx = self.idx  # Use filtered indices
        while idx:  # Continue until all indices are processed
            batch = []  # Initialize an empty batch
            n_nodes = 0  # Initialize node count for the current batch
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes:
                next_idx, idx = idx[0], idx[1:]  # Pop the first index
                n_nodes += self.node_counts[next_idx]  # Update node count
                batch.append(next_idx)  # Add index to the current batch
            self.batches.append(batch)  # Append the formed batch

    def __len__(self):
        if not self.batches:
            self._form_batches()  # Form batches if they are not already formed
        return len(self.batches)  # Return the number of batches

    def __iter__(self):
        if not self.batches:
            self._form_batches()  # Form batches if they are not already formed
        for batch in self.batches:
            yield batch  # Yield each batch for iteration


class ProteinGraphDataset(data.Dataset):
    """
    A map-style `torch.utils.data.Dataset` which transforms JSON/dictionary-style
    protein structures into featurized protein graphs as described in the
    manuscript.

    Returned graphs are of type `torch_geometric.data.Data` with attributes
    -x          alpha carbon coordinates, shape [n_nodes, 3]
    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
    -name       name of the protein structure, string
    -node_s     node scalar features, shape [n_nodes, 6]
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, 32]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    -edge_index edge indices, shape [2, n_edges]
    -mask       node mask, `False` for nodes with missing data that are excluded from message passing

    Portions from https://github.com/jingraham/neurips19-graph-protein-design.

    :param data_list: JSON/dictionary-style protein dataset as described in README.md.
    :param num_positional_embeddings: number of positional embeddings
    :param top_k: number of edges to draw per node (as destination node)
    :param device: if "cuda", will do preprocessing on the GPU
    """

    def __init__(
        self,
        dataset,
        index,
        args,
        task_list,
        num_positional_embeddings=16,
        top_k=50,
        num_rbf=16,
        device="cpu",
    ):

        super(ProteinGraphDataset, self).__init__()

        self.dataset = {}  # Initialize an empty dataset dictionary
        index = set(index)  # Convert index list to a set for fast lookup
        for i, ID in enumerate(dataset):  # Enumerate through the dataset
            if i in index:  # If index is in the specified indices
                self.dataset[ID] = dataset[ID]  # Add the data to the dataset dictionary
        self.IDs = list(self.dataset.keys())  # Store all dataset IDs

        self.fasta_file = args.fasta_file  # Store path to the FASTA file
        self.output_prottrans = (
            args.prottrans_output_path
        )  # Store output path for ProtTrans features
        self.output_esm2 = args.esm2_output_path  # Store output path for ESM2 features
        self.proc_pdb_output_path = (
            args.proc_pdb_output_path
        )  # Store output path for processed PDB files
        self.pdb_dir = args.pdb_dir  # Store directory for PDB files
        self.output_dssp = args.dssp_output_path  # Store output path for DSSP features
        self.task_list = task_list  # Store the list of tasks to perform

        self.top_k = top_k  # Store the number of edges per node
        self.num_rbf = num_rbf  # Store the number of RBF channels
        self.num_positional_embeddings = (
            num_positional_embeddings  # Store the number of positional embeddings
        )
        self.device = device  # Store the device for computations ('cpu' or 'cuda')
        self.node_counts = [
            len(self.dataset[ID][0]) for ID in self.IDs
        ]  # Store the node count for each protein in the dataset

        self.letter_to_num = {
            "C": 4,
            "D": 3,
            "S": 15,
            "Q": 5,
            "K": 11,
            "I": 9,
            "P": 14,
            "T": 16,
            "F": 13,
            "A": 0,
            "G": 7,
            "H": 8,
            "E": 6,
            "L": 10,
            "R": 1,
            "W": 17,
            "V": 19,
            "N": 2,
            "Y": 18,
            "M": 12,
            "X": 20,
        }  # Dictionary mapping amino acids to numerical indices
        self.num_to_letter = {
            v: k for k, v in self.letter_to_num.items()
        }  # Reverse mapping from numerical indices to amino acids

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        return self._featurize_as_graph(idx)

    def _featurize_as_graph(self, idx):
        # Featurize the protein at the given index as a graph
        name = self.IDs[idx]  # Get the protein name from the ID list
        with torch.no_grad():  # Disable gradient computation for featurization
            coords = np.load(
                self.proc_pdb_output_path + name + ".npy", allow_pickle=True
            )  # Load the coordinates from a .npy file

            coords = torch.as_tensor(
                coords, device=self.device, dtype=torch.float32
            )  # Convert coordinates to a tensor

            seq = torch.as_tensor(
                [self.letter_to_num[aa] for aa in self.dataset[name]],
                device=self.device,
                dtype=torch.long,
            )  # Convert sequence to tensor of integer indices
            mask = torch.isfinite(
                coords.sum(dim=(1, 2))
            )  # Create a mask for nodes with valid coordinates
            coords[~mask] = np.inf  # Set coordinates of masked nodes to infinity

            X_ca = coords[:, 1]  # Extract alpha carbon coordinates
            edge_index = torch_cluster.knn_graph(
                X_ca, k=self.top_k
            )  # Create a k-nearest neighbors graph

            pos_embeddings = self._positional_embeddings(
                edge_index
            )  # Compute positional embeddings for edges
            E_vectors = (
                X_ca[edge_index[0]] - X_ca[edge_index[1]]
            )  # Calculate edge vectors

            rbf = _rbf(
                E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device
            )  # Compute RBF embeddings for edge lengths
            geo_edge_feat = self._get_geo_edge_feat(
                X_ca, edge_index
            )  # Compute geometric features for edges
            dihedrals = self._dihedrals(
                coords
            )  # Compute dihedral angles as node features
            orientations = self._orientations(
                X_ca
            )  # Compute backbone orientations as node features
            sidechains = self._sidechains(
                coords
            )  # Compute sidechain orientations as node features

            prottrans_feat = torch.tensor(
                np.load(self.output_prottrans + name + ".npy", allow_pickle=True)
            )  # Load ProtTrans features
            esm_feat = torch.tensor(
                np.load(self.output_esm2 + name + ".npy", allow_pickle=True)
            )  # Load ESM2 features
            dssp = torch.tensor(
                np.load(self.output_dssp + name + "_dssp.npy", allow_pickle=True)
            )  # Load DSSP features

            node_s = torch.cat([dihedrals, prottrans_feat, esm_feat, dssp], dim=-1).to(
                torch.float32
            )  # Combine node scalar features
            node_v = torch.cat(
                [orientations, sidechains.unsqueeze(-2)], dim=-2
            )  # Combine node vector features

            edge_s = torch.cat(
                [rbf, pos_embeddings], dim=-1
            )  # Combine edge scalar features
            edge_v = _normalize(E_vectors).unsqueeze(
                -2
            )  # Normalize edge vectors as edge vector features

            node_s, node_v, edge_s, edge_v = map(
                torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
            )  # Replace NaNs in features with zeros

            y_true = np.stack(
                prot2annot[name][self.task_list]
            )  # Stack annotations for the tasks
            y = torch.as_tensor(
                y_true, device=self.device, dtype=torch.float32
            )  # Convert annotations to a tensor

        # Create a data object with all the computed features
        data = torch_geometric.data.Data(
            x=X_ca,
            seq=seq,
            name=name,
            node_s=node_s,
            node_v=node_v,
            edge_s=edge_s,
            edge_v=edge_v,
            edge_index=edge_index,
            mask=mask,
            y=y,
        )

        return data  # Return the data object

    def _dihedrals(self, X, eps=1e-7):
        # Compute dihedral angles as node features

        X = torch.reshape(
            X[:, :3], [3 * X.shape[0], 3]
        )  # Reshape coordinates for backbone atoms
        dX = X[1:] - X[:-1]  # Compute differences between consecutive backbone atoms
        U = _normalize(dX, dim=-1)  # Normalize differences
        u_2 = U[:-2]  # Get vectors for dihedral calculation
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(
            torch.cross(u_2, u_1), dim=-1
        )  # Compute cross products for normals
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)  # Compute dot product of normals
        cosD = torch.clamp(
            cosD, -1 + eps, 1 - eps
        )  # Clamp values to avoid numerical issues
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(
            cosD
        )  # Compute dihedral angles

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])  # Pad angles to match node count
        D = torch.reshape(D, [-1, 3])  # Reshape angles
        # Lift angle representations to the circle
        D_features = torch.cat(
            [torch.cos(D), torch.sin(D)], 1
        )  # Convert angles to sin and cos features
        return D_features  # Return dihedral features

    def _positional_embeddings(
        self, edge_index, num_embeddings=None, period_range=[2, 1000]
    ):
        # Compute positional embeddings for edges
        num_embeddings = (
            num_embeddings or self.num_positional_embeddings
        )  # Set number of embeddings if not provided
        d = edge_index[0] - edge_index[1]  # Compute distances between connected nodes

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )  # Compute frequencies for positional encoding
        angles = d.unsqueeze(-1) * frequency  # Compute angles for encoding
        E = torch.cat(
            (torch.cos(angles), torch.sin(angles)), -1
        )  # Combine sin and cos of angles
        return E  # Return positional embeddings

    def _orientations(self, X):
        # Compute backbone orientations as node features
        forward = _normalize(X[1:] - X[:-1])  # Normalize forward differences
        backward = _normalize(X[:-1] - X[1:])  # Normalize backward differences
        forward = F.pad(forward, [0, 0, 0, 1])  # Pad forward vector
        backward = F.pad(backward, [0, 0, 1, 0])  # Pad backward vector
        return torch.cat(
            [forward.unsqueeze(-2), backward.unsqueeze(-2)], -2
        )  # Combine orientations

    def _sidechains(self, X):
        # Compute sidechain orientations as node features
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]  # Extract N, CA, and C atoms
        c, n = _normalize(c - origin), _normalize(
            n - origin
        )  # Normalize C and N vectors
        bisector = _normalize(c + n)  # Compute bisector of C and N
        perp = _normalize(torch.cross(c, n))  # Compute perpendicular vector
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(
            2 / 3
        )  # Compute sidechain vector
        return vec  # Return sidechain orientation vector

    def _get_geo_edge_feat(self, X_ca, edge_index):
        # Compute geometric features for edges
        u = torch.ones_like(X_ca)  # Initialize unit vectors for backbone
        u[1:] = X_ca[1:] - X_ca[:-1]  # Compute backbone vectors
        u = F.normalize(u, dim=-1)  # Normalize backbone vectors
        b = torch.ones_like(X_ca)  # Initialize backbone difference vectors
        b[:-1] = u[:-1] - u[1:]  # Compute differences between backbone vectors
        b = F.normalize(b, dim=-1)  # Normalize difference vectors
        n = torch.ones_like(X_ca)  # Initialize normal vectors
        n[:-1] = torch.cross(u[:-1], u[1:])  # Compute normals using cross products
        n = F.normalize(n, dim=-1)  # Normalize normals

        local_frame = torch.stack(
            [b, n, torch.cross(b, n)], dim=-1
        )  # Stack vectors to form local reference frames

        node_j, node_i = edge_index  # Extract source and destination node indices
        t = F.normalize(X_ca[node_j] - X_ca[node_i], dim=-1)  # Compute edge vectors
        t = torch.einsum(
            "ijk,ij->ik", local_frame[node_i], t
        )  # Project edge vectors into local frames
        # r = torch.sum(local_frame[node_i] * local_frame[node_j], dim=1)
        r = torch.matmul(
            local_frame[node_i].transpose(-1, -2), local_frame[node_j]
        )  # Compute rotations between frames
        Q = self._quaternions(r)  # Convert rotations to quaternions

        return torch.cat([t, Q], dim=-1)  # Return concatenated geometric edge features

    def _quaternions(self, R):
        # Convert rotation matrices to quaternions
        diag = torch.diagonal(R, dim1=-2, dim2=-1)  # Extract diagonal elements
        Rxx, Ryy, Rzz = diag.unbind(-1)  # Unbind diagonals
        magnitudes = 0.5 * torch.sqrt(
            torch.abs(
                1
                + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1)
            )
        )  # Compute magnitudes for quaternion components
        _R = lambda i, j: R[
            :, i, j
        ]  # Helper function to extract elements from rotation matrices
        signs = torch.sign(
            torch.stack(
                [_R(2, 1) - _R(1, 2), _R(0, 2) - _R(2, 0), _R(1, 0) - _R(0, 1)], -1
            )
        )  # Compute signs for quaternion components
        xyz = signs * magnitudes  # Compute xyz components of quaternions
        # The relu enforces a non-negative trace
        w = (
            torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0
        )  # Compute w component
        Q = torch.cat((xyz, w), -1)  # Concatenate quaternion components
        Q = F.normalize(Q, dim=-1)  # Normalize quaternions

        return Q  # Return quaternions
