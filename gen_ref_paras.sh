#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# raw
python gen_aligned_ref.py --ref_file data/wmt24/json_output_ja_zh/raw/ref_A.jsonl --segmenter spacy --src_lang ja --task_lang zh --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_ja_zh/raw/ref_A.jsonl --segmenter ersatz --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_en_es/raw/ref_A.jsonl --segmenter spacy --src_lang en --task_lang es --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_en_es/raw/ref_A.jsonl --segmenter ersatz --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_en_de/raw/ref_A.jsonl --segmenter spacy --src_lang en --task_lang de --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_en_de/raw/ref_A.jsonl --segmenter ersatz --proc_device cuda -v --overlap 20 

# drop_tgt
python gen_aligned_ref.py --ref_file data/wmt24/json_output_ja_zh/drop_tgt/ref_A.jsonl --segmenter spacy --src_lang ja --task_lang zh --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_ja_zh/drop_tgt/ref_A.jsonl --segmenter ersatz --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_en_es/drop_tgt/ref_A.jsonl --segmenter spacy --src_lang en --task_lang es --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_en_es/drop_tgt/ref_A.jsonl --segmenter ersatz --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_en_de/drop_tgt/ref_A.jsonl --segmenter spacy --src_lang en --task_lang de --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_en_de/drop_tgt/ref_A.jsonl --segmenter ersatz --proc_device cuda -v --overlap 20

# drop_src
python gen_aligned_ref.py --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl --segmenter spacy --src_lang ja --task_lang zh --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl --segmenter ersatz --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl --segmenter spacy --src_lang en --task_lang es --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl --segmenter ersatz --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl --segmenter spacy --src_lang en --task_lang de --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl --segmenter ersatz --proc_device cuda -v --overlap 20

# merge
python gen_aligned_ref.py --ref_file data/wmt24/json_output_ja_zh/merge/ref_A.jsonl --segmenter spacy --src_lang ja --task_lang zh --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_ja_zh/merge/ref_A.jsonl --segmenter ersatz --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_en_es/merge/ref_A.jsonl --segmenter spacy --src_lang en --task_lang es --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_en_es/merge/ref_A.jsonl --segmenter ersatz --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_en_de/merge/ref_A.jsonl --segmenter spacy --src_lang en --task_lang de --proc_device cuda -v --overlap 20
python gen_aligned_ref.py --ref_file data/wmt24/json_output_en_de/merge/ref_A.jsonl --segmenter ersatz --proc_device cuda -v --overlap 20
