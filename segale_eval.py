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

import json
from collections import defaultdict
import os
import re
import json
import csv
import spacy
import torch
import random
import argparse
import numpy as np
import pandas as pd
import tempfile
import subprocess
import unicodedata
from multiprocessing import Pool
from tqdm import tqdm
import datetime
from typing import Optional
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from comet import download_model, load_from_checkpoint

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """
    Set the global random seed for reproducibility.

    Args:
        seed (int): Random seed (default is 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Metrics Computation
# -----------------------------------------------------------------------------

import glob
def clear_specific_lock_files():
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    pattern = os.path.join(cache_dir, "*cache_huggingface_datasets_json_default*.lock")
    lock_files = glob.glob(pattern)
    
    for lock_file in lock_files:
        try:
            os.remove(lock_file)
            print(f"Removed file: {lock_file}")
        except Exception as e:
            print(f"Error removing {lock_file}: {e}")

def run_comet_evaluation(aggregated_windows):
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    zero_score_windows = []
    none_windows = []
    comet_scores = []
    data = []
    # Write each window on a separate line
    for idx, window in enumerate(aggregated_windows):
        src, ref, mt = window
        if src and ref and mt:
            data.append({"src": src, "mt": mt, "ref": ref})
        elif (mt and not src and not ref) or (src and ref and not mt):
            zero_score_windows.append(idx)
        else:
            print(f"WARNING: Empty reference-based window detected. Pass.")
            none_windows.append(idx)
    if data:
        model_outputs = model.predict(data, batch_size=8, gpus=1)
        comet_scores = model_outputs.scores  # list of float scores
    
    # Insert zero scores for windows that had missing scores
    for idx in zero_score_windows:
        comet_scores.insert(idx, 0.0)
    for idx in none_windows:
        comet_scores.insert(idx, -1)
    return comet_scores

def run_comet_qe_evaluation(aggregated_windows):
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)
    zero_score_windows = []
    none_windows = []
    comet_qe_scores = []
    data = []

    # Write each window on a separate line
    for idx, window in enumerate(aggregated_windows):
        src, mt = window
        if src and mt:
            data.append({"src": src, "mt": mt})
        elif (mt and not src) or (src and not mt):
            zero_score_windows.append(idx)
        else:
            print(f"WARNING: Empty reference-free window detected. Pass.")
            none_windows.append(idx)
    if data:
        model_outputs = model.predict(data, batch_size=8, gpus=1)
        comet_qe_scores = model_outputs.scores
    
    # Insert zero scores for windows that had missing scores
    for idx in zero_score_windows:
        comet_qe_scores.insert(idx, 0.0)
    for idx in none_windows:
        comet_qe_scores.insert(idx, -1)
    return comet_qe_scores

def run_metricx_evaluation(aggregated_windows):
    zero_score_windows = []
    none_windows = [] 
    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".jsonl") as metricx_ref_input, \
         tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".jsonl") as metricx_ref_output:
        
        for idx, window in enumerate(aggregated_windows):
            src, ref, mt = window
            if src and ref and mt:
                json_obj = {"source": src, "hypothesis": mt, "reference": ref}
                metricx_ref_input.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
            elif (mt and not src and not ref) or (src and ref and not mt):
                zero_score_windows.append(idx)
            else:
                print(f"WARNING: Empty reference-based window detected. Pass.")
                none_windows.append(idx)
        
        metricx_ref_input.flush()
        metricx_ref_input_name = metricx_ref_input.name
        metricx_ref_output.flush()
        metricx_ref_output_name = metricx_ref_output.name

        metricx_ref_command = [
            "python", "-m", "metricx24.predict",
            "--tokenizer", "google/mt5-large",
            "--model_name_or_path", "google/metricx-24-hybrid-large-v2p6",
            "--max_input_length", "1536",
            "--batch_size", "1",
            "--input_file", metricx_ref_input_name,
            "--output_file", metricx_ref_output_name,
        ]
    
        result_metricx_ref = subprocess.run(metricx_ref_command, stdout=subprocess.PIPE, text=True)
        print(result_metricx_ref.stdout)

        metricx_ref_scores = []
        with open(metricx_ref_output_name, 'r', encoding='utf-8') as f:
            for line in f:
                data_line = json.loads(line)
                prediction = data_line.get("prediction", 0)
                metricx_ref_scores.append(float(prediction))

    clear_specific_lock_files()

    # Insert zero scores for windows that had missing scores
    for idx in zero_score_windows:
        metricx_ref_scores.insert(idx, 25)
    for idx in none_windows:
        metricx_ref_scores.insert(idx, -1)
    return metricx_ref_scores



def run_metricx_qe_evaluation(aggregated_windows):
    zero_score_windows = []
    none_windows = []
    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".jsonl") as metricx_qe_input, \
         tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".jsonl") as metricx_qe_output:
        
        for idx, window in enumerate(aggregated_windows):
            src, mt = window
            if src and mt:
                json_obj = {"source": src, "hypothesis": mt, "reference": ""}
                metricx_qe_input.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
            elif (mt and not src) or (src and not mt):
                zero_score_windows.append(idx)
            else:
                print(f"WARNING: Empty reference-free window detected. Pass.")
                none_windows.append(idx)
        
        metricx_qe_input.flush()
        metricx_qe_input_name = metricx_qe_input.name

        metricx_qe_output.flush()
        metricx_qe_output_name = metricx_qe_output.name

        metricx_qe_command = [
            "python", "-m", "metricx24.predict",
            "--tokenizer", "google/mt5-large",
            "--model_name_or_path", "google/metricx-24-hybrid-large-v2p6",
            "--max_input_length", "1536",
            "--batch_size", "1",
            "--input_file", metricx_qe_input_name,
            "--output_file", metricx_qe_output_name,
            "--qe"
        ]
        result_metricx_qe = subprocess.run(metricx_qe_command, stdout=subprocess.PIPE, text=True)
        print(result_metricx_qe.stdout)

        metricx_qe_scores = []
        with open(metricx_qe_output_name, 'r', encoding='utf-8') as f:
            for line in f:
                data_line = json.loads(line)
                prediction = data_line.get("prediction", 0)
                metricx_qe_scores.append(float(prediction))

    clear_specific_lock_files()
    
    # Insert zero scores for windows that had missing scores
    for idx in zero_score_windows:
        metricx_qe_scores.insert(idx, 25)
    for idx in none_windows:
        metricx_qe_scores.insert(idx, -1)
    return metricx_qe_scores


# -----------------------------------------------------------------------------
# main function
# -----------------------------------------------------------------------------

def evaluate_and_aggregate(input_file, eval_output_file, aggregated_output_file):
    """
    Reads an aligned_results.jsonl file and computes evaluation scores using the provided functions.
    Then:
      1. Attaches the evaluation scores (comet, comet-qe, metricx, metricx-qe) to each JSON object
         and writes the result to eval_output_file.
      2. Aggregates objects by doc_id (sorting by seg_id and joining src, ref, tgt with '\n'),
         computes average scores, and writes the aggregated objects to aggregated_output_file.
    """

    # Read the original data
    with open(input_file, 'r', encoding='utf-8') as fin:
        results = [json.loads(line) for line in fin if line.strip()]

    # Prepare windows for evaluation
    comet_windows = []       # tuples of (src, ref, tgt)
    metricx_windows = []     # tuples of (src, ref, tgt)
    comet_qe_windows = []    # tuples of (src, tgt)
    metricx_qe_windows = []  # tuples of (src, tgt)

    for item in results:
        src = item.get('src', '')
        ref = item.get('ref', '')
        tgt = item.get('tgt', '')
        comet_windows.append((src, ref, tgt))
        metricx_windows.append((src, ref, tgt))
        comet_qe_windows.append((src, tgt))
        metricx_qe_windows.append((src, tgt))
        
    # Call evaluation functions to get scores (order must match the order in results)
    comet_scores      = run_comet_evaluation(comet_windows)
    comet_qe_scores   = run_comet_qe_evaluation(comet_qe_windows)
    metricx_scores    = run_metricx_evaluation(metricx_windows)
    metricx_qe_scores = run_metricx_qe_evaluation(metricx_qe_windows)

    # Attach the scores to each JSON object
    for idx, item in enumerate(results):
        item['comet']      = comet_scores[idx]
        item['comet-qe']   = comet_qe_scores[idx]
        item['metricx']    = metricx_scores[idx]
        item['metricx-qe'] = metricx_qe_scores[idx]

    # Write the updated objects with evaluation scores to eval_output_file
    with open(eval_output_file, 'w', encoding='utf-8') as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Aggregate objects by doc_id
    grouped = defaultdict(list)
    for item in results:
        doc_id = item.get('doc_id')
        grouped[doc_id].append(item)

    aggregated_results = []
    for doc_id, items in grouped.items():
        # Sort items by seg_id (defaulting to 0 if not present)
        items_sorted = sorted(items, key=lambda x: int(x.get('seg_id', '0').split('_')[0]) if isinstance(x.get('seg_id'), str) and '_' in x.get('seg_id', '') else x.get('seg_id', 0))
        sys_id = items_sorted[0].get('sys_id', '')

        src_agg = "\n".join(item.get('src', '') for item in items_sorted)
        ref_agg = "\n".join(item.get('ref', '') for item in items_sorted)
        tgt_agg = "\n".join(item.get('tgt', '') for item in items_sorted)

        # Filter valid scores (score >= 0)
        valid_comet = [item['comet'] for item in items_sorted if item['comet'] >= 0]
        valid_comet_qe = [item['comet-qe'] for item in items_sorted if item['comet-qe'] >= 0]
        valid_metricx = [item['metricx'] for item in items_sorted if item['metricx'] >= 0]
        valid_metricx_qe = [item['metricx-qe'] for item in items_sorted if item['metricx-qe'] >= 0]

        # Calculate averages, avoid division by zero
        comet_avg = sum(valid_comet) / len(valid_comet) if valid_comet else 0
        comet_qe_avg = sum(valid_comet_qe) / len(valid_comet_qe) if valid_comet_qe else 0
        metricx_avg = sum(valid_metricx) / len(valid_metricx) if valid_metricx else 0
        metricx_qe_avg = sum(valid_metricx_qe) / len(valid_metricx_qe) if valid_metricx_qe else 0

        # Count valid segments (>=0) and misaligned segments (comet == 0)
        total_seg = len(valid_comet)
        misaligned_seg = valid_comet.count(0)

        aggregated_obj = {
            "doc_id": doc_id,
            "sys_id": sys_id,
            "src": src_agg,
            "ref": ref_agg,
            "tgt": tgt_agg,
            "comet": comet_avg,
            "comet-qe": comet_qe_avg,
            "metricx": metricx_avg,
            "metricx-qe": metricx_qe_avg,
            "total_seg": total_seg,
            "misaligned_seg": misaligned_seg
        }
        aggregated_results.append(aggregated_obj)

    # Write the aggregated results to aggregated_output_file
    with open(aggregated_output_file, 'w', encoding='utf-8') as fout:
        for obj in aggregated_results:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

# Example command: python segale_eval.py --input_file data/wmt24/json_output_ja_zh/raw/GPT-4/aligned_spacy_GPT-4.jsonl
def main():
    parser = argparse.ArgumentParser(description="Evaluate aligned_results.jsonl and aggregate results by doc_id")
    parser.add_argument("--input_file", type=str, default="aligned_results.jsonl",
                        help="Path to the input aligned_results.jsonl file")
    args = parser.parse_args()

    print("\n\n\n##########################################")
    print(f"Current execution: {args}")
    print("##########################################\n\n\n")
    
    folder_path = os.path.dirname(args.input_file)

    evaluate_and_aggregate(
        input_file=args.input_file,
        eval_output_file= os.path.join(folder_path, ("eval_" + str(args.input_file).split('/')[-1])),
        aggregated_output_file= os.path.join(folder_path, ("result_" + str(args.input_file).split('/')[-1])),
    )

if __name__ == '__main__':
    main()
