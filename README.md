# Query-Driven Attribution Framework for Iterative Layout Generation (LayoutFlow XAI)

This repository is the official implementation of the 2026 paper “Query-Driven Attribution Framework for Iterative Layout Generation”, where we apply the framework to the LayoutFlow method and choose Integrated Gradients as the attribution method.
It is based on the official implementation of the LayoutFlow 2024 paper ([project page](https://julianguerreiro.github.io/layoutflow/) | [paper](https://arxiv.org/pdf/2403.18187)).

## Setup and Run

For everything related to setup and running this project, please consult `README_ORIGINAL.md`, which largely reflects the LayoutFlow README.

## Configuration Related to XAI

In `conf/test.yaml`, we defined the following configuration options:

- `seed`: The seed used across the entire codebase.
- `explain`: If `True`, the generated layouts will be explained (a list of `instance_ids` must be provided).
- `ig_steps`: The number of steps used for Integrated Gradients.
- `influence_mode`: The attribution aggregation scheme (`grouped_all` for N-EAM, `grouped_psc` for FG-EAM and `per_xy` for SA-XY).
- `target_attr`: The target attribute, which can be: `x`, `y`, `w`, `h`, `position`, `size`, `geometry` or `category`.
- `instance_ids`: The dataset instance IDs to explain; for each instance, one target element is selected at random and explained.

With the `calculate_stats` option in `conf/train.yaml`, the training step will be disabled and replaced by the calculations needed to construct the Integrated Gradients baseline we use (which differs across datasets).

## Results

For all output produced across our experiments—run with different combinations of configuration settings—see our [Hugging Face LayoutFlow XAI Dataset repository](https://huggingface.co/datasets/vaarga/layoutflow-xai).

Special thanks to Patricia Claudia Moțiu and Marc Eduard Frîncu for their collaboration.
