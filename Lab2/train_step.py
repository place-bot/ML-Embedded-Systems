"""Exercise: complete the three optimization operations."""


def train_step(model, inputs, targets, loss_fn, optimizer):
    optimizer.zero_grad()
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
    correct = (logits.detach().argmax(dim=1) == targets)
    return loss.detach().item(), correct.sum().item()
