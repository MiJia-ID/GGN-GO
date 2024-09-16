import torch
import torch.nn as nn
import torch.distributed as dist

import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        # Forward pass: gather tensors from all processes
        ctx.save_for_backward(input)  # Save input tensor for backward pass
        output = [
            torch.zeros_like(input) for _ in range(dist.get_world_size())
        ]  # Create a list to hold gathered tensors from all processes
        dist.all_gather(
            output, input
        )  # Gather tensors from all processes into the list
        return tuple(output)  # Return gathered tensors as a tuple

    @staticmethod
    def backward(ctx, *grads):
        # Backward pass: distribute gradients back to each process
        (input,) = ctx.saved_tensors  # Retrieve saved input tensor
        grad_out = torch.zeros_like(input)  # Create a tensor to hold the gradient
        grad_out[:] = grads[
            dist.get_rank()
        ]  # Set gradient for the current process using its rank
        return grad_out  # Return the gradient tensor


class NT_Xent(nn.Module):
    """Normalized Temperature-scaled Cross Entropy (NT-Xent) Loss."""

    def __init__(self, batch_size, temperature, world_size):
        # Initialize NT-Xent loss with batch size, temperature, and world size
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        # Create a mask to exclude positive pairs from the set of negative samples
        N = 2 * batch_size * world_size  # Total number of samples across all processes
        mask = torch.ones((N, N), dtype=bool)  # Initialize a mask of ones
        mask = mask.fill_diagonal_(
            0
        )  # Set diagonal elements to zero (self-correlations)
        for i in range(batch_size * world_size):  # Iterate over each sample
            mask[i, batch_size + i] = (
                0  # Mask correlations between positive pairs (i and batch_size + i)
            )
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        Forward pass for NT-Xent loss calculation.
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N . 1) augmented examples within a minibatch as negative examples.
        """
        N = (
            2 * self.batch_size * self.world_size
        )  # Calculate the total number of samples in all processes

        z = torch.cat(
            (z_i, z_j), dim=0
        )  # Concatenate positive pairs along the batch dimension
        if self.world_size > 1:  # If distributed training with multiple processes
            z = torch.cat(
                GatherLayer.apply(z), dim=0
            )  # Gather all tensors from all processes

        sim = (
            self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        )  # Compute similarity matrix and scale by temperature

        sim_i_j = torch.diag(
            sim, self.batch_size * self.world_size
        )  # Extract positive similarities (i, j)
        sim_j_i = torch.diag(
            sim, -self.batch_size * self.world_size
        )  # Extract positive similarities (j, i)

        # With distributed training, each GPU gets N samples, resulting in 2xNxN matrix
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            N, 1
        )  # Combine positive similarities into a single tensor
        negative_samples = sim[self.mask].reshape(
            N, -1
        )  # Extract negative samples using the mask

        labels = (
            torch.zeros(N).to(positive_samples.device).long()
        )  # Create labels (all zeros for positives at index 0)
        logits = torch.cat(
            (positive_samples, negative_samples), dim=1
        )  # Concatenate positive and negative samples for logits
        loss = self.criterion(
            logits, labels
        )  # Calculate cross-entropy loss between logits and labels
        loss /= N  # Normalize loss by the number of samples
        return loss  # Return the loss value
