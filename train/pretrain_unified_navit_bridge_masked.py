"""Bridge-necessity training entrypoint.

Thin shim around `pretrain_unified_navit.main` that swaps in
`BridgeMaskedPackedDataset` for `PackedDataset` so the original trainer
file is left untouched.

Run with the same argv as `pretrain_unified_navit.py`; configure the
masking via env vars BRIDGE_MASK_* (see BridgeMaskedPackedDataset).
"""

from __future__ import annotations

import sys

from data.bridge_masked_packed_dataset import BridgeMaskedPackedDataset

# Importing the trainer module *binds* its `PackedDataset` symbol; we
# override that binding before main() runs so the dataset constructed inside
# main() is our subclass. dataset_base.PackedDataset is also overridden in
# case some helper in main looks it up there.
import train.pretrain_unified_navit as _trainer
import data.dataset_base as _ds

_trainer.PackedDataset = BridgeMaskedPackedDataset
_ds.PackedDataset = BridgeMaskedPackedDataset


if __name__ == "__main__":
    _trainer.main()
