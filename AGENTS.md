# Agent Notes

- Run cartpole training jobs outside the sandbox when GPU/MPS is needed. The sandbox hides
  MPS from PyTorch on this machine, while the same command outside the sandbox can see
  `mps:0`.
- Do not run training jobs with `--device cpu`; training is expected to use CUDA/MPS.
- When optimizing training or evaluation behavior, do not attempt to improve results by
  changing the random seed.
