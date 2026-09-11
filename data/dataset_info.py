# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import (
    UnifiedEditIterableDataset,
    SpatialReasoningIterableDataset,
    VisualOnlyThinkingIterableDataset,
    BridgeMaskedSpatialReasoningIterableDataset,
    BridgeMaskedVisualOnlyThinkingIterableDataset,
)
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'spatial_reasoning': SpatialReasoningIterableDataset,
    'visual_only_thinking': VisualOnlyThinkingIterableDataset,
    'bridge_masked_spatial_reasoning': BridgeMaskedSpatialReasoningIterableDataset,
    'bridge_masked_visual_only_thinking': BridgeMaskedVisualOnlyThinkingIterableDataset,
}


DATASET_INFO = {
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
		},
    },
    'spatial_reasoning': {
        'infinigen_spatial': {
            'data_dir': '/path/to/scratch/VisualCoT/training_data/spatial_reasoning',
            'num_files': 14,
            'num_total_samples': 1429,
            "parquet_info_path": '/path/to/scratch/VisualCoT/training_data/spatial_reasoning/parquet_info.json',
        },
        'mix_panorama_visual': {
            'data_dir': '/path/to/scratch/VisualCoT/training_data/mix_panorama_sft/panorama_visual',
            'num_total_samples': 3000,  # TODO: update after data generation completes (check parquet_info.json)
            'parquet_info_path': '/path/to/scratch/VisualCoT/training_data/mix_panorama_sft/panorama_visual/parquet_info.json',
        },
    },
    'visual_only_thinking': {
        'infinigen_spatial_visual_only': {
            'data_dir': '/path/to/scratch/VisualCoT/training_data/spatial_reasoning_visual_only',
            'parquet_info_path': '/path/to/scratch/VisualCoT/training_data/spatial_reasoning_visual_only/parquet_info.json',
        },
        'training_data_mix_all_rotation_balance': {
            'data_dir': '/path/to/scratch/infinigen/training_data_mix_all_rotation/visual_only',
            'parquet_info_path': '/path/to/scratch/infinigen/training_data_mix_all_rotation/visual_only/parquet_info.json',
            'num_total_samples': 9223,
        },
    },
    'bridge_masked_spatial_reasoning': {
        'infinigen_spatial': {
            'data_dir': '/path/to/scratch/VisualCoT/training_data/spatial_reasoning',
            'num_files': 14,
            'num_total_samples': 1429,
            'parquet_info_path': '/path/to/scratch/VisualCoT/training_data/spatial_reasoning/parquet_info.json',
        },
        'mix_panorama_visual': {
            'data_dir': '/path/to/scratch/VisualCoT/training_data/mix_panorama_sft/panorama_visual',
            'num_total_samples': 3000,
            'parquet_info_path': '/path/to/scratch/VisualCoT/training_data/mix_panorama_sft/panorama_visual/parquet_info.json',
        },
    },
    'bridge_masked_visual_only_thinking': {
        'infinigen_spatial_visual_only': {
            'data_dir': '/path/to/scratch/VisualCoT/training_data/spatial_reasoning_visual_only',
            'parquet_info_path': '/path/to/scratch/VisualCoT/training_data/spatial_reasoning_visual_only/parquet_info.json',
        },
        'training_data_mix_all_rotation_balance': {
            'data_dir': '/path/to/scratch/infinigen/training_data_mix_all_rotation/visual_only',
            'parquet_info_path': '/path/to/scratch/infinigen/training_data_mix_all_rotation/visual_only/parquet_info.json',
            'num_total_samples': 9223,
        },
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'your_data_path/bagel_example/vlm/images',
			'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
        'infinigen_spatial_text_only': {
            'data_dir': '/path/to/scratch/spatial_collab_dataset/scenes',
            'jsonl_path': '/path/to/scratch/VisualCoT/training_data/spatial_reasoning_scence_graph_text_only/text_only_thinking.jsonl',
            'num_total_samples': 10000,
        },
        'mix_sft_text_only': {
            'data_dir': '/path/to/scratch/datasets/ViewFusion-traindata',
            'jsonl_path': '/path/to/scratch/VisualCoT/training_data/mix_panorama_sft/sft_text_only/text_only_thinking.jsonl',
            'num_total_samples': 2000,
        },
        'mix_all_understanding': {
            'data_dir': '/network/scratch',
            'jsonl_path': '/path/to/scratch/infinigen/training_data_mix_all/understanding/understanding.jsonl',
            'num_total_samples': 9223,
        },
        'training_data_point_matching': {
            'data_dir': '/network/scratch',
            'jsonl_path': '/path/to/scratch/infinigen/training_data_point_matching/no_thinking/no_thinking.jsonl',
            'num_total_samples': 911,
        },
        'training_data_point_matching_matterport': {
            'data_dir': '/network/scratch',
            'jsonl_path': '/path/to/scratch/matterport/training_data_point_matching_matterport/no_thinking/no_thinking.jsonl',
            'num_total_samples': 1500,
        },
        'training_data_mix_all_balance': {
            'data_dir': '/network/scratch',
            'jsonl_path': '/path/to/scratch/infinigen/training_data_mix_all_balance/no_thinking/no_thinking.jsonl',
            'num_total_samples': 7921,
        },
        'training_data_matterport_rotation': {
            'data_dir': '/network/scratch',
            'jsonl_path': '/path/to/scratch/infinigen/training_data_matterport_rotation/no_thinking/no_thinking.jsonl',
            'num_total_samples': 9000,
        },
    },
}
