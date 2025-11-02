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
from tqdm import tqdm
import datetime
from typing import Optional, List, Dict, Any
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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

def run_metricx_evaluation(aggregated_windows):
    zero_score_windows = []
    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".jsonl") as metricx_ref_input, \
         tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".jsonl") as metricx_ref_output:
        
        for idx, window in enumerate(aggregated_windows):
            if window is None:
                zero_score_windows.append(idx)
            else:
                src, ref, mt = window
                if src and mt:
                    json_obj = {"source": src, "hypothesis": mt, "reference": ref}
                    metricx_ref_input.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                else:
                    zero_score_windows.append(idx)
        
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

    return metricx_ref_scores

def run_metricx_qe_evaluation(aggregated_windows):
    zero_score_windows = []
    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".jsonl") as metricx_qe_input, \
         tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".jsonl") as metricx_qe_output:
        
        for idx, window in enumerate(aggregated_windows):
            if window is None:
                zero_score_windows.append(idx)
            else:
                src, mt = window
                if src and mt:
                    json_obj = {"source": src, "hypothesis": mt, "reference": ""}
                    metricx_qe_input.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                else:
                    zero_score_windows.append(idx)
        
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

    return metricx_qe_scores

# -----------------------------------------------------------------------------
# File Evaluation Function 
# -----------------------------------------------------------------------------
def read_jsonl(file_path):
    """Read a JSONL file and return a list of JSON objects."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def merge_system_entries(entries):
    """
    Merge multiple system JSONL entries by doc_id.
    For each doc_id, sort the entries by seg_id and concatenate the src and tgt (MT) texts.
    """
    merged = {}
    for entry in entries:
        doc_id = entry["doc_id"]
        if doc_id not in merged:
            merged[doc_id] = {
                "doc_id": doc_id,
                "sys_id": entry.get("sys_id", ""),
                "src_list": [],
                "tgt_list": [],
                "seg_ids": []
            }
        merged[doc_id]["src_list"].append(entry["src"])
        merged[doc_id]["tgt_list"].append(entry["tgt"])
        merged[doc_id]["seg_ids"].append(entry["seg_id"])
    
    # Sort entries by seg_id and merge the texts
    for doc_id, info in merged.items():
        sorted_indices = sorted(range(len(info["seg_ids"])), key=lambda i: info["seg_ids"][i])
        src_merged = " ".join([info["src_list"][i] for i in sorted_indices])
        tgt_merged = " ".join([info["tgt_list"][i] for i in sorted_indices])
        info["src"] = src_merged
        info["tgt"] = tgt_merged
    return merged

def merge_ref_entries(entries):
    """
    Merge reference JSONL (e.g., ref_A.jsonl) entries by doc_id.
    Concatenate the tgt fields for each doc_id to form the final reference text.
    """
    merged = {}
    for entry in entries:
        doc_id = entry["doc_id"]
        if doc_id not in merged:
            merged[doc_id] = {
                "doc_id": doc_id,
                "ref_list": [],
                "seg_ids": []
            }
        merged[doc_id]["ref_list"].append(entry["tgt"])
        merged[doc_id]["seg_ids"].append(entry["seg_id"])
    
    for doc_id, info in merged.items():
        sorted_indices = sorted(range(len(info["seg_ids"])), key=lambda i: info["seg_ids"][i] if isinstance(info["seg_ids"][i], int) else int(info["seg_ids"][i].split('_')[0]))
        ref_merged = " ".join([info["ref_list"][i] for i in sorted_indices])
        info["ref"] = ref_merged
    return merged


def combine_system_ref(system_merged, ref_merged):
    """
    Combine the merged system and reference data by doc_id.
    If a corresponding doc_id is not found in the reference, the ref field is set to an empty string.
    Returns a list where each element contains the system and reference texts.
    """
    combined = []
    for doc_id, sys_info in system_merged.items():
        combined.append({
            "doc_id": doc_id,
            "sys_id": sys_info["sys_id"],
            "src": sys_info["src"],
            "tgt": sys_info["tgt"],
            "ref": ref_merged.get(doc_id, {}).get("ref", "")
        })
    return combined

def extract_lang(system_file: str) -> str:
    m = re.search(r'(en[-_](?:de|es)|ja[-_]zh)', system_file, re.IGNORECASE)
    if m:
        return m.group(1).replace("_", "-").lower()
    return "unknown"

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("metricx24/spiece.model")
def count_tokens(text: str) -> int:
    tokens = sp.encode(text)
    return len(tokens)

# ------------------------------ drop n operation  ------------------------------ #
def drop_segments(entries, drop_n):
    groups = {}
    for entry in entries:
        doc_id = entry["doc_id"]
        groups.setdefault(doc_id, []).append(entry)
    new_entries = []
    dropped_segments = {}
    for doc_id, segs in groups.items():
        try:
            segs_sorted = sorted(segs, key=lambda x: int(x["seg_id"]) if isinstance(x["seg_id"], int)
                                 else int(x["seg_id"].split('_')[0]))
        except Exception as e:
            segs_sorted = sorted(segs, key=lambda x: x["seg_id"])
        if len(segs_sorted) < drop_n + 1:
            continue
        kept = segs_sorted[:len(segs_sorted) - drop_n]
        dropped = segs_sorted[len(segs_sorted) - drop_n:]
        new_entries.extend(kept)
        dropped_segments[doc_id] = [str(d["seg_id"]) for d in dropped]
    return new_entries, dropped_segments

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(
        description="Preliminary experiment with drop processing and metricx evaluation using input folder"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory (e.g., wmt24/en-de) containing ref_A.jsonl and system JSONL files")
    parser.add_argument("--scenario", type=str, choices=["over", "under"], required=True,
                        help="Drop scenario: 'over' (drop on src & ref) or 'under' (drop on system's tgt)")
    parser.add_argument("--output_dir", type=str, default="data/preli",
                        help="Base output directory (default: preli)")
    args = parser.parse_args()

    lang = extract_lang(args.input_dir)
    print(f"Detected language pair: {lang}")
    
    ref_file = os.path.join(args.input_dir, "ref_A.jsonl")
    if not os.path.exists(ref_file):
        print(f"Error: Reference file ref_A.jsonl not found in {args.input_dir}")
        return
    ref_entries_all = read_jsonl(ref_file)

    all_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    system_files = [f for f in all_files if os.path.basename(f) != "ref_A.jsonl"]
    if not system_files:
        print(f"No system JSONL files found in {args.input_dir}")
        return
    
    dropped_seg_ids = None
    for sys_file in system_files:
        print(f"\nProcessing system file: {sys_file}")
        system_entries_all = read_jsonl(sys_file)

        for drop_n in range(5):
            print(f"\n  Processing drop_n = {drop_n} for scenario '{args.scenario}'")
            if args.scenario == "over":
                system_entries_src, dropped_seg_ids = drop_segments(system_entries_all, drop_n)
                valid_doc_ids_src = set(entry["doc_id"] for entry in system_entries_src)

                ref_entries_drop, dropped_seg_ids = drop_segments(ref_entries_all, drop_n)
                valid_doc_ids_ref = set(entry["doc_id"] for entry in ref_entries_drop)

                if valid_doc_ids_src != valid_doc_ids_ref:
                    print("Warning: The doc_ids from src and ref are not identical!")
                    print("src doc_ids:", valid_doc_ids_src)
                    print("ref doc_ids:", valid_doc_ids_ref)
                else:
                    print("Check passed: src and ref doc_ids are identical.")

                valid_doc_ids = valid_doc_ids_src

                system_entries_src = [entry for entry in system_entries_src if entry["doc_id"] in valid_doc_ids]
                system_entries_tgt = [entry for entry in system_entries_all if entry["doc_id"] in valid_doc_ids]
                ref_entries_drop = [entry for entry in ref_entries_drop if entry["doc_id"] in valid_doc_ids]

                merged_src = merge_system_entries(system_entries_src)
                merged_tgt = merge_system_entries(system_entries_tgt)
                system_merged = {}
                for doc_id in merged_src.keys():
                    if doc_id in merged_tgt:
                        system_merged[doc_id] = {
                            "doc_id": doc_id,
                            "sys_id": merged_src[doc_id].get("sys_id", ""),
                            "src": merged_src[doc_id]["src"],
                            "tgt": merged_tgt[doc_id]["tgt"]
                        }
                merged_ref = merge_ref_entries(ref_entries_drop)
            else:
                system_entries_tgt, dropped_seg_ids = drop_segments(system_entries_all, drop_n)
                valid_doc_ids = set(entry["doc_id"] for entry in system_entries_tgt)
                system_entries_src = [entry for entry in system_entries_all if entry["doc_id"] in valid_doc_ids]
                merged_src = merge_system_entries(system_entries_src)
                merged_tgt = merge_system_entries(system_entries_tgt)
                system_merged = {}
                for doc_id in merged_src.keys():
                    if doc_id in merged_tgt:
                        system_merged[doc_id] = {
                            "doc_id": doc_id,
                            "sys_id": merged_src[doc_id].get("sys_id", ""),
                            "src": merged_src[doc_id]["src"],
                            "tgt": merged_tgt[doc_id]["tgt"]
                        }
                ref_entries_valid = [entry for entry in ref_entries_all if entry["doc_id"] in valid_doc_ids]
                merged_ref = merge_ref_entries(ref_entries_valid)

            combined_docs = combine_system_ref(system_merged, merged_ref)
            print(f"    After drop_n={drop_n}, {len(combined_docs)} doc(s) remain.")
    
            # -------------------------- < 1024 tokens. --------------------------#
            original_count = len(combined_docs)
            dropped_doc_ids = set()
            kept_docs = []
            for doc in combined_docs:
                src_tokens = count_tokens(doc["src"])
                tgt_tokens = count_tokens(doc["tgt"])
                ref_tokens = count_tokens(doc["ref"])
                total_tokens = src_tokens + tgt_tokens + ref_tokens
                if total_tokens > 1024:
                    dropped_doc_ids.add(doc["doc_id"])
                else:
                    kept_docs.append(doc)
    
            drop_count = len(dropped_doc_ids)
            drop_percentage = (drop_count / original_count) * 100 if original_count > 0 else 0
            print(f"drop {drop_count} doc_id，ratio: {drop_percentage:.2f}%")
    
            # -------------------------- update dataset --------------------------#
            combined_docs = kept_docs
            system_merged_filtered = {}
            for doc_id, entry in system_merged.items():
                if doc_id not in dropped_doc_ids:
                    system_merged_filtered[doc_id] = entry
            system_merged = system_merged_filtered
    
            # -------------------------- evaluate dataset --------------------------#
            system_windows = []
            for doc_id, entry in system_merged.items():
                system_windows.append((entry["src"], entry["tgt"]))
            aggregated_metricx_qe_scores = run_metricx_qe_evaluation(system_windows)

            combined_windows = []
            for doc in combined_docs:
                combined_windows.append((doc["src"], doc["ref"], doc["tgt"]))
            aggregated_metricx_scores = run_metricx_evaluation(combined_windows)
    
            # -------------------------- save evaluation result --------------------------#
            final_results = []
            for i, doc in enumerate(combined_docs):
                final_results.append({
                    "doc_id": doc["doc_id"],
                    "sys_id": doc["sys_id"],
                    "src": doc["src"],
                    "tgt": doc["tgt"],
                    "ref": doc["ref"],
                    "metricx": aggregated_metricx_scores[i],
                    "metricx-qe": aggregated_metricx_qe_scores[i]
                })
            out_dir = os.path.join(args.output_dir + '/' + args.scenario, lang, f"drop_{drop_n}")
            os.makedirs(out_dir, exist_ok=True)
            base_filename = os.path.basename(sys_file)
            output_path = os.path.join(out_dir, base_filename)
            with open(output_path, "w", encoding="utf-8") as fout:
                for res in final_results:
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            print(f"    Output saved to: {output_path}")

            # saving deleted seg_id tp txt
            drop_record_path = os.path.join(out_dir, base_filename.replace(".jsonl", ".txt"))
            with open(drop_record_path, 'w', encoding='utf-8') as f:
                for doc_id, seg_ids in dropped_seg_ids.items():
                    line = f"{doc_id}: {', '.join(seg_ids)}\n"
                    f.write(line)
            print(f"    Dropped seg_id saved to: {drop_record_path}")

if __name__ == "__main__":
    main()

