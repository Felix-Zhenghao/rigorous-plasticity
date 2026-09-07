# Models and loss regularizers

The architecture builders return ordinary PyTorch modules with a fixed final
projection. The learner owns its optimizer, update counter and independent RNG.
Inference construction uses the resolved architecture dictionary saved in every
learner state; auxiliary heads and method histories are not inference parameters.
No learner receives task, chunk or epoch information.

## Sources checked

The following code was inspected at the revisions linked by the implementation
plan, alongside the local paper PDFs:

| Implementation | Source | Convention |
| --- | --- | --- |
| L2 and L2 Init | `skumar9876/L2_Init`, `3f86d26ff06c20a39c556dfc43a3ecd844d5c55e`, `agents/agents.py` | The penalty has the explicit factor 1/2. L2 Init retains each parameter's own initial value. Normalization affine parameters are excluded by default. |
| InFeR | `timoklein/infer`, `bfcab6c76d9d06806736b60a7dd8a61db7fd68da`, `src/agent.py` and `infer_dqn.py` | Frozen initial features and scalar auxiliary heads; target scaling only on the frozen predictions. This repository is a third-party port. |
| Spectral | *Learning Continually by Spectral Regularization*, Eq. (1), Appendix A.7 | Differentiable spectral estimate, convolution flattening, gain targets of one and additive targets of zero. The untested third-party repository is not used as a numerical reference. |
| Feature norm | *Disentangling the Causes of Plasticity Loss in Neural Networks*, §4.1 | Feature sites and reductions follow the explicit testbed specification because the paper does not specify them fully. |

InFeR defaults to summing the squared scalar-head errors before averaging over
examples, matching the paper equation. `head_reduction: mean` matches the inspected
port; equal coefficients then differ by `num_heads`. The auxiliary heads are one
linear projection with independent rows and are registered once in the optimizer.
The frozen extractor includes hidden readout layers. Its final supervised
projection is removed because the reference only predicts auxiliary outputs.
No RL hyperparameter is silently supplied as a supervised default.

Spectral power vectors are persistent and detached. The scalar `u @ W @ v` is
computed with gradients enabled. Convolutions use `[out, in * kh * kw]`, position
tables use matrices, and CLS tokens use the vector/bias penalty. The latter
embedding convention and the elementwise squared normalization-gain penalty are
explicit testbed choices. Residual LN stores delta and regularizes effective
gain `1 + delta` toward one.

## Architecture choices

MLPs expose `hidden`, `head_hidden`, and the final `head`. Each hidden layer owns
its linear, normalization, activation and dropout modules. Hidden head layers
belong to the feature extractor. ResNet-18 has four stages of two BasicBlocks,
with a configurable small or ImageNet stem. LayerNorm replaces every BN,
including projection shortcuts. Channel LN permutes NCHW to NHWC; full-feature
LN has separate affine parameters for every CHW coordinate. NaP can add a
normalization immediately before the residual sum activation.

ViT supports convolutional or linear patch embeddings, pre-norm blocks, optional
patch normalization for linear embeddings, separate Q/K/V projections, and
explicit attention, embedding and feedforward dropout. Mean pooling averages
patch tokens, excluding the CLS token. Named paper presets supply reported
capacities; they do not claim identical paper training or initialization.

The default initialization is Kaiming uniform with activation-specific gain,
zero affine-layer biases, a linear readout gain, unit normalization gains and
normal embedding values with standard deviation 0.02. Other explicit presets
are Kaiming normal, Xavier uniform, and PyTorch's uniform affine initialization.
Every parameter saves its distribution, shape and original fan values. Sampling
replacement entries reuses those full-layer fans and never reconstructs a model.
Learners report main and auxiliary trainable counts and persistent method-state
bytes; the latter includes frozen references, anchors and detached power vectors,
but excludes optimizer tensors. Reference input buffers are counted separately.

The feature-norm sites are `hidden.N`, `head_hidden.N` and `head_input` for MLP;
ResNet additionally offers `stem`, `pooled`, and
`stages.S.B.activation1`/`activation2`. Only requested feature tensors are kept.
`head_input` always means the tensor entering the final output projection.

## Verification

`tests/test_models_methods.py` checks all supported pairs for one-step training
and identical continued training after save/restore, fixed full-output semantics,
L2 gradients and frozen anchors, feature reductions, trainable/frozen InFeR
components, spectral gradients against exact SVD, LN replacement and residual
gains, LeakyReLU coverage, sliced initialization fans, deterministic prediction
and dropout restoration, configuration rejection, and stationary fitting.
These checks verify operations, not empirical resistance to plasticity loss.
