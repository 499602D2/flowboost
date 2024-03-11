# Optimizer interfaces
FlowBoost supports any arbitrary optimizer backend. Adding support for one entails writing an interface, which sits between FlowBoost's generic `Optimizer`, and the typically much more complex backend of an optimization framework, such as `skopt` or `Ax`.

## Requirements
