from typing import Any, Callable


class MultiOutputLoss:
    def __init__(
            self, 
            n_outputs: int, 
            loss: Callable
        ) -> None:
        self.n_outputs = n_outputs
        self.loss = loss

    def __call__(self, preds, targets) -> Any:
        assert targets.shape[-1] == self.n_outputs

        losses = []
        for k in range(self.n_outputs):
            losses.append(self.loss(preds[k], targets[:,k]))

        total = sum(losses)
        return total
