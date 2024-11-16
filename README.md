# Talos Library

<p align="center">
<img src="image.webp" align="center" alt="image" height="250em"/>

Library for dealing with ML stuff for One-shot imitation learning.
</p>



## Quick start

To get started, clone this repository to your machine:
```bash
git clone https://github.com/StealthyPanda/taloslib.git
```

In the root directory of the package (`taloslib`), run `pip install`:

> **_NOTE:_**  Make sure you have activated your desired python environment, if using virtual environments before running pip install.

```bash
cd taloslib
pip install -e .
```

This installs the package as an editable package. To update the package, you can simply run `git pull` to update the code.


## Overview

There are 3 main submodules:

- `talos.datapipe` : contains all dataset creation, loading and saving related stuff.
- `talos.training` : contains various model training related stuff.
- `talos.models` : contains different model architectures, stuff for loading pretrain models etc.

`talos.utils` contains general helpful stuff related to GPUs info etc.

You can go through `example.ipynb` to get a more comprehensive tutorial.
