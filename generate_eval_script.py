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

import os

BASE_DIR = "data/wmt24"
SH_FILE = "run_eval.sh"

# python generate_eval_script.py
# bash run_eval.sh

IGNORE_FILES = {"ref_A.jsonl", "ref_B.jsonl", "spacy_ref_A.jsonl", "ersatz_ref_A.jsonl"}
REFERENCE = "ref_A.jsonl" # "spacy_ref_A.jsonl", "ersatz_ref_A.jsonl"
SEGMENTER = "spacy"
EXP_FOLDER = ["raw"] # "raw", "merge", "drop_src", "drop_tgt"
VERBOSE = "-v"

with open(SH_FILE, "w") as f:
    f.write("#!/bin/bash\n\n")

    for lang_pair in os.listdir(BASE_DIR):
        lang_pair_path = os.path.join(BASE_DIR, lang_pair)
        if not os.path.isdir(lang_pair_path):
            continue

        lang_src, lang_tgt = lang_pair.split("_")[-2:]

        for subfolder in EXP_FOLDER:
            sub_dir = os.path.join(lang_pair_path, subfolder)
            if not os.path.isdir(sub_dir):
                continue

            ref_A_path = os.path.join(sub_dir, REFERENCE)
            if not os.path.exists(ref_A_path):
                continue

            # step 1 & 2:
            for filename in os.listdir(sub_dir):
                if not filename.endswith(".jsonl") or filename in IGNORE_FILES:
                    continue

                jsonl_path = os.path.join(sub_dir, filename)
                base_name = os.path.splitext(filename)[0]
                out_dir = os.path.join(sub_dir, base_name)
                aligned_path = os.path.join(out_dir, "aligned_" + SEGMENTER + "_" + base_name + ".jsonl")

                # step 1: gen aligned src-mt-ref
                step2_cmd = f"""python wmtAlign.py --system_file {jsonl_path}\\
    --ref_file {os.path.join(sub_dir, REFERENCE)} \\
    --segmenter {SEGMENTER} \\
    --task_lang {lang_tgt} --proc_device cuda {VERBOSE}\n"""

                # step 2: evaluation
                step3_cmd = f"""python wmtEval.py --input_file {aligned_path} \n"""
                f.write(f"# Step 1 & 2: {lang_pair}/{subfolder}/{base_name}\n{step2_cmd}{step3_cmd}\n")
                # f.write(f"# Step 1: {lang_pair}/{subfolder}/{base_name}\n{step2_cmd}")
                # f.write(f"# Step 2: {lang_pair}/{subfolder}/{base_name}\n{step3_cmd}\n")

print(f" Done! Generate`{SH_FILE}`。You can run `bash {SH_FILE}` for evaluation")

