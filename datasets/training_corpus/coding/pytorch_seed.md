# PyTorch Seed Knowledge

PyTorch is a local tensor and neural network library. A common neural network
training loop creates a model, defines a loss function, creates an optimizer,
runs a forward pass, calculates loss, calls backward, and steps the optimizer.

```python
optimizer.zero_grad()
output = model(batch)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```
