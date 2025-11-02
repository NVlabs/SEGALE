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

# Step 1 & 2: json_output_ja_zh/drop_src/Claude-3.5
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/Claude-3.5.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/Claude-3.5/aligned_spacy_Claude-3.5.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/DLUT_GTCOM
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/DLUT_GTCOM.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/DLUT_GTCOM/aligned_spacy_DLUT_GTCOM.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/Team-J
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/Team-J.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/Team-J/aligned_spacy_Team-J.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/Gemini-1.5-Pro
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/Gemini-1.5-Pro.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/Gemini-1.5-Pro/aligned_spacy_Gemini-1.5-Pro.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/GPT-4
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/GPT-4.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/GPT-4/aligned_spacy_GPT-4.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/Unbabel-Tower70B
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/Unbabel-Tower70B.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/Unbabel-Tower70B/aligned_spacy_Unbabel-Tower70B.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/MSLC
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/MSLC.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/MSLC/aligned_spacy_MSLC.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/Llama3-70B
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/Llama3-70B.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/Llama3-70B/aligned_spacy_Llama3-70B.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/ONLINE-A
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/ONLINE-A.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/ONLINE-A/aligned_spacy_ONLINE-A.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/Phi-3-Medium
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/Phi-3-Medium.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/Phi-3-Medium/aligned_spacy_Phi-3-Medium.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/IKUN-C
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/IKUN-C.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/IKUN-C/aligned_spacy_IKUN-C.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/NTTSU
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/NTTSU.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/NTTSU/aligned_spacy_NTTSU.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/ONLINE-B
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/ONLINE-B.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/ONLINE-B/aligned_spacy_ONLINE-B.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/CommandR-plus
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/CommandR-plus.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/CommandR-plus/aligned_spacy_CommandR-plus.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/Aya23
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/Aya23.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/Aya23/aligned_spacy_Aya23.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/ONLINE-G
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/ONLINE-G.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/ONLINE-G/aligned_spacy_ONLINE-G.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/IOL_Research
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/IOL_Research.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/IOL_Research/aligned_spacy_IOL_Research.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/Mistral-Large
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/Mistral-Large.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/Mistral-Large/aligned_spacy_Mistral-Large.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/ONLINE-W
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/ONLINE-W.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/ONLINE-W/aligned_spacy_ONLINE-W.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/UvA-MT
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/UvA-MT.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/UvA-MT/aligned_spacy_UvA-MT.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/CycleL
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/CycleL.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/CycleL/aligned_spacy_CycleL.jsonl 

# Step 1 & 2: json_output_ja_zh/drop_src/IKUN
segale-align --system_file data/wmt24/json_output_ja_zh/drop_src/IKUN.jsonl\
    --ref_file data/wmt24/json_output_ja_zh/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang zh --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_ja_zh/drop_src/IKUN/aligned_spacy_IKUN.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/Claude-3.5
segale-align --system_file data/wmt24/json_output_en_es/drop_src/Claude-3.5.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/Claude-3.5/aligned_spacy_Claude-3.5.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/TSU-HITs
segale-align --system_file data/wmt24/json_output_en_es/drop_src/TSU-HITs.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/TSU-HITs/aligned_spacy_TSU-HITs.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/Occiglot
segale-align --system_file data/wmt24/json_output_en_es/drop_src/Occiglot.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/Occiglot/aligned_spacy_Occiglot.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/Gemini-1.5-Pro
segale-align --system_file data/wmt24/json_output_en_es/drop_src/Gemini-1.5-Pro.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/Gemini-1.5-Pro/aligned_spacy_Gemini-1.5-Pro.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/GPT-4
segale-align --system_file data/wmt24/json_output_en_es/drop_src/GPT-4.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/GPT-4/aligned_spacy_GPT-4.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/Unbabel-Tower70B
segale-align --system_file data/wmt24/json_output_en_es/drop_src/Unbabel-Tower70B.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/Unbabel-Tower70B/aligned_spacy_Unbabel-Tower70B.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/TranssionMT
segale-align --system_file data/wmt24/json_output_en_es/drop_src/TranssionMT.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/TranssionMT/aligned_spacy_TranssionMT.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/MSLC
segale-align --system_file data/wmt24/json_output_en_es/drop_src/MSLC.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/MSLC/aligned_spacy_MSLC.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/Llama3-70B
segale-align --system_file data/wmt24/json_output_en_es/drop_src/Llama3-70B.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/Llama3-70B/aligned_spacy_Llama3-70B.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/ONLINE-A
segale-align --system_file data/wmt24/json_output_en_es/drop_src/ONLINE-A.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/ONLINE-A/aligned_spacy_ONLINE-A.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/Phi-3-Medium
segale-align --system_file data/wmt24/json_output_en_es/drop_src/Phi-3-Medium.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/Phi-3-Medium/aligned_spacy_Phi-3-Medium.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/IKUN-C
segale-align --system_file data/wmt24/json_output_en_es/drop_src/IKUN-C.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/IKUN-C/aligned_spacy_IKUN-C.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/ONLINE-B
segale-align --system_file data/wmt24/json_output_en_es/drop_src/ONLINE-B.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/ONLINE-B/aligned_spacy_ONLINE-B.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/CommandR-plus
segale-align --system_file data/wmt24/json_output_en_es/drop_src/CommandR-plus.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/CommandR-plus/aligned_spacy_CommandR-plus.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/Aya23
segale-align --system_file data/wmt24/json_output_en_es/drop_src/Aya23.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/Aya23/aligned_spacy_Aya23.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/NVIDIA-NeMo
segale-align --system_file data/wmt24/json_output_en_es/drop_src/NVIDIA-NeMo.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/NVIDIA-NeMo/aligned_spacy_NVIDIA-NeMo.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/ONLINE-G
segale-align --system_file data/wmt24/json_output_en_es/drop_src/ONLINE-G.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/ONLINE-G/aligned_spacy_ONLINE-G.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/Dubformer
segale-align --system_file data/wmt24/json_output_en_es/drop_src/Dubformer.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/Dubformer/aligned_spacy_Dubformer.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/IOL_Research
segale-align --system_file data/wmt24/json_output_en_es/drop_src/IOL_Research.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/IOL_Research/aligned_spacy_IOL_Research.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/Mistral-Large
segale-align --system_file data/wmt24/json_output_en_es/drop_src/Mistral-Large.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/Mistral-Large/aligned_spacy_Mistral-Large.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/ONLINE-W
segale-align --system_file data/wmt24/json_output_en_es/drop_src/ONLINE-W.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/ONLINE-W/aligned_spacy_ONLINE-W.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/CycleL
segale-align --system_file data/wmt24/json_output_en_es/drop_src/CycleL.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/CycleL/aligned_spacy_CycleL.jsonl 

# Step 1 & 2: json_output_en_es/drop_src/IKUN
segale-align --system_file data/wmt24/json_output_en_es/drop_src/IKUN.jsonl\
    --ref_file data/wmt24/json_output_en_es/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang es --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_es/drop_src/IKUN/aligned_spacy_IKUN.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/Claude-3.5
segale-align --system_file data/wmt24/json_output_en_de/drop_src/Claude-3.5.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/Claude-3.5/aligned_spacy_Claude-3.5.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/TSU-HITs
segale-align --system_file data/wmt24/json_output_en_de/drop_src/TSU-HITs.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/TSU-HITs/aligned_spacy_TSU-HITs.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/CUNI-NL
segale-align --system_file data/wmt24/json_output_en_de/drop_src/CUNI-NL.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/CUNI-NL/aligned_spacy_CUNI-NL.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/Occiglot
segale-align --system_file data/wmt24/json_output_en_de/drop_src/Occiglot.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/Occiglot/aligned_spacy_Occiglot.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/Gemini-1.5-Pro
segale-align --system_file data/wmt24/json_output_en_de/drop_src/Gemini-1.5-Pro.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/Gemini-1.5-Pro/aligned_spacy_Gemini-1.5-Pro.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/GPT-4
segale-align --system_file data/wmt24/json_output_en_de/drop_src/GPT-4.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/GPT-4/aligned_spacy_GPT-4.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/Unbabel-Tower70B
segale-align --system_file data/wmt24/json_output_en_de/drop_src/Unbabel-Tower70B.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/Unbabel-Tower70B/aligned_spacy_Unbabel-Tower70B.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/TranssionMT
segale-align --system_file data/wmt24/json_output_en_de/drop_src/TranssionMT.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/TranssionMT/aligned_spacy_TranssionMT.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/MSLC
segale-align --system_file data/wmt24/json_output_en_de/drop_src/MSLC.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/MSLC/aligned_spacy_MSLC.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/AIST-AIRC
segale-align --system_file data/wmt24/json_output_en_de/drop_src/AIST-AIRC.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/AIST-AIRC/aligned_spacy_AIST-AIRC.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/Llama3-70B
segale-align --system_file data/wmt24/json_output_en_de/drop_src/Llama3-70B.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/Llama3-70B/aligned_spacy_Llama3-70B.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/ONLINE-A
segale-align --system_file data/wmt24/json_output_en_de/drop_src/ONLINE-A.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/ONLINE-A/aligned_spacy_ONLINE-A.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/Phi-3-Medium
segale-align --system_file data/wmt24/json_output_en_de/drop_src/Phi-3-Medium.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/Phi-3-Medium/aligned_spacy_Phi-3-Medium.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/IKUN-C
segale-align --system_file data/wmt24/json_output_en_de/drop_src/IKUN-C.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/IKUN-C/aligned_spacy_IKUN-C.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/ONLINE-B
segale-align --system_file data/wmt24/json_output_en_de/drop_src/ONLINE-B.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/ONLINE-B/aligned_spacy_ONLINE-B.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/CommandR-plus
segale-align --system_file data/wmt24/json_output_en_de/drop_src/CommandR-plus.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/CommandR-plus/aligned_spacy_CommandR-plus.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/Aya23
segale-align --system_file data/wmt24/json_output_en_de/drop_src/Aya23.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/Aya23/aligned_spacy_Aya23.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/NVIDIA-NeMo
segale-align --system_file data/wmt24/json_output_en_de/drop_src/NVIDIA-NeMo.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/NVIDIA-NeMo/aligned_spacy_NVIDIA-NeMo.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/ONLINE-G
segale-align --system_file data/wmt24/json_output_en_de/drop_src/ONLINE-G.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/ONLINE-G/aligned_spacy_ONLINE-G.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/Dubformer
segale-align --system_file data/wmt24/json_output_en_de/drop_src/Dubformer.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/Dubformer/aligned_spacy_Dubformer.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/IOL_Research
segale-align --system_file data/wmt24/json_output_en_de/drop_src/IOL_Research.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/IOL_Research/aligned_spacy_IOL_Research.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/Mistral-Large
segale-align --system_file data/wmt24/json_output_en_de/drop_src/Mistral-Large.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/Mistral-Large/aligned_spacy_Mistral-Large.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/ONLINE-W
segale-align --system_file data/wmt24/json_output_en_de/drop_src/ONLINE-W.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/ONLINE-W/aligned_spacy_ONLINE-W.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/CycleL2
segale-align --system_file data/wmt24/json_output_en_de/drop_src/CycleL2.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/CycleL2/aligned_spacy_CycleL2.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/CycleL
segale-align --system_file data/wmt24/json_output_en_de/drop_src/CycleL.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/CycleL/aligned_spacy_CycleL.jsonl 

# Step 1 & 2: json_output_en_de/drop_src/IKUN
segale-align --system_file data/wmt24/json_output_en_de/drop_src/IKUN.jsonl\
    --ref_file data/wmt24/json_output_en_de/drop_src/ref_A.jsonl \
    --segmenter spacy \
    --task_lang de --proc_device cuda -v
segale-eval --input_file data/wmt24/json_output_en_de/drop_src/IKUN/aligned_spacy_IKUN.jsonl 

