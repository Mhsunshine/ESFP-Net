import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
            dist.all_gather(output, input)
            return tuple(output)
        else:
            # If no distributed environment, return the input itself as output
            return (input,)
    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)

        # dist.reduce_scatter(grad_out, list(grads))
        # grad_out.div_(dist.get_world_size())

        if dist.is_available() and dist.is_initialized():
            grad_out[:] = grads[dist.get_rank()]
        else:
            # If no distributed environment, just return the gradient for the input
            grad_out = grads[0]

        return grad_out