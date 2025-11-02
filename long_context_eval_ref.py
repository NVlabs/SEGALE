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


def segment_sentences_by_ersatz(text: str) -> list:
    with tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8", suffix=".txt") as temp_in:
        temp_in.write(text)
        temp_in.flush()
        input_filename = temp_in.name
    output_filename = input_filename + ".segmented"    
    subprocess.run(["ersatz", "--input", input_filename, "--output", output_filename], check=True)    
    with open(output_filename, "r", encoding="utf-8") as f:
        segmented_text = f.read()
    os.remove(input_filename)
    os.remove(output_filename)
    sentences = [line.strip() for line in segmented_text.splitlines() if line.strip()]
    return sentences

def segment_sentences_by_spacy(text: str) -> list:
    segmented_sentences = []
    paragraphs = text.split('\n')
    for paragraph in paragraphs:
        if paragraph.strip():
            doc = mt_seg(paragraph)
            for sent in doc.sents:
                segmented_sentences.append(sent.text.strip())
    return segmented_sentences



def generate_overlap_and_embedding(text: str) -> tuple:
    """
    Generate overlap and embedding data from text using temporary files.

    Args:
        text (str): Input text.

    Returns:
        tuple: (overlap_content (str), embeddings_content (bytes))
    """
    with tempfile.NamedTemporaryFile(delete=True, mode="w+", encoding="utf-8", suffix=".txt") as txt_file:
        txt_file.write(text)
        txt_file.flush()
        txt_filename = txt_file.name
        overlaps_file = txt_filename + ".overlaps"
        embed_file = txt_filename + ".emb"

        # Generate overlap data
        subprocess.run(["./overlap.py", "-i", txt_filename, "-o", overlaps_file, "-n", "10"], check=True)
        # Generate embedding data
        subprocess.run(" ".join(["$LASER/tasks/embed/embed.sh", overlaps_file, embed_file]),
                       shell=True, check=True)

        with open(embed_file, "rb") as f:
            embeddings_content = f.read()
        with open(overlaps_file, "r", encoding="utf-8") as f:
            overlap_content = f.read()

    for need_to_del_file in [overlaps_file, embed_file]:
        try:
            os.remove(need_to_del_file)
            print(f"Removed file: {need_to_del_file}")
        except Exception as e:
            print(f"Error removing {need_to_del_file}: {e}")

    return overlap_content, embeddings_content


def compute_alignment_stats(alignment_results: list) -> tuple:
    """
    Compute the average alignment cost (ignoring zero-cost alignments) and the zero-cost ratio.

    Args:
        alignment_results (list): List of alignment result strings in the format "[src]:[tgt]:cost".

    Returns:
        tuple: (average_cost (float), zero_cost_ratio (float))
    """
    costs = []
    zero_cost_count = 0

    for entry in alignment_results:
        try:
            cost = float(entry.split(":")[-1])
            if cost == 0.0:
                zero_cost_count += 1
            else:
                costs.append(cost)
        except ValueError:
            continue

    avg_cost = sum(costs) / len(costs) if costs else 0.0
    zero_cost_ratio = zero_cost_count / len(alignment_results) if alignment_results else 0.0

    return avg_cost, zero_cost_ratio


def run_vecalign_explore(src_text: str, tgt_text: str, src_overlap: str, tgt_overlap: str,
                         src_embed: bytes, tgt_embed: bytes) -> list:
    """
    Explore the best vector alignment parameters and return the best alignments.

    Args:
        src_text (str): Source text.
        tgt_text (str): Target text.
        src_overlap (str): Overlap data for the source.
        tgt_overlap (str): Overlap data for the target.
        src_embed (bytes): Embedding data for the source.
        tgt_embed (bytes): Embedding data for the target.

    Returns:
        list: Parsed best alignments as a list of tuples [(src_indices, tgt_indices), ...].
    """
    del_percentile_frac = 0.2
    step_size = 0.005
    prev_zero_cost_ratio = None
    prev_avg_cost = None

    best_avg_cost = float('inf')
    best_del_percentile_frac = del_percentile_frac
    best_zero_cost_ratio = 0.0
    best_alignments = []

    first_flag = True
    with tempfile.NamedTemporaryFile(delete=True, mode="w+", encoding="utf-8", suffix=".txt") as src_file, \
         tempfile.NamedTemporaryFile(delete=True, mode="w+", encoding="utf-8", suffix=".txt") as tgt_file, \
         tempfile.NamedTemporaryFile(delete=True, mode="w+", encoding="utf-8", suffix=".overlaps") as src_overlap_file, \
         tempfile.NamedTemporaryFile(delete=True, mode="w+", encoding="utf-8", suffix=".overlaps") as tgt_overlap_file, \
         tempfile.NamedTemporaryFile(delete=True, mode="wb", suffix=".emb") as src_embed_file, \
         tempfile.NamedTemporaryFile(delete=True, mode="wb", suffix=".emb") as tgt_embed_file:

        src_file.write(src_text)
        src_file.flush()
        tgt_file.write(tgt_text)
        tgt_file.flush()

        src_overlap_file.write(src_overlap)
        src_overlap_file.flush()
        tgt_overlap_file.write(tgt_overlap)
        tgt_overlap_file.flush()

        src_embed_file.write(src_embed)
        src_embed_file.flush()
        tgt_embed_file.write(tgt_embed)
        tgt_embed_file.flush()

        while del_percentile_frac > 0:
            result = subprocess.run(
                [
                    "./vecalign.py",
                    "--alignment_max_size", "8",
                    "--del_percentile_frac", str(del_percentile_frac),
                    "--src", src_file.name,
                    "--tgt", tgt_file.name,
                    "--src_embed", src_overlap_file.name, src_embed_file.name,
                    "--tgt_embed", tgt_overlap_file.name, tgt_embed_file.name,
                ],
                stdout=subprocess.PIPE,
                text=True,
            )

            output_lines = result.stdout.strip().split("\n")
            avg_cost, zero_cost_ratio = compute_alignment_stats(output_lines)
            print(f"del_percentile_frac: {del_percentile_frac:.3f} | Avg Cost: {avg_cost:.6f} | Zero-Cost Ratio: {zero_cost_ratio:.2%}")

            if first_flag:
                first_flag = False

            if prev_zero_cost_ratio is not None and prev_zero_cost_ratio != 0 and (zero_cost_ratio / prev_zero_cost_ratio) > 1.5:
                print(f"Stopping exploration: Zero-cost ratio increased sharply at {del_percentile_frac:.3f}")
                break
            elif prev_zero_cost_ratio is not None and (
                (zero_cost_ratio - prev_zero_cost_ratio) > 0.15 or
                avg_cost > prev_avg_cost or
                avg_cost < 0.3 or zero_cost_ratio > 0.7
            ):
                print(f"Stopping exploration: Zero-cost ratio increased sharply at {del_percentile_frac:.3f}")
                break
            else:
                if avg_cost < best_avg_cost:
                    best_avg_cost = avg_cost
                    best_del_percentile_frac = del_percentile_frac
                    best_zero_cost_ratio = zero_cost_ratio
                    best_alignments = output_lines

            prev_zero_cost_ratio = zero_cost_ratio
            prev_avg_cost = avg_cost
            del_percentile_frac -= step_size

    # Parse the best alignments
    parsed_alignments = []
    for line in best_alignments:
        if line:
            src_part, tgt_part, _ = line.split(":")
            src_indices = list(map(int, src_part.strip("[]").split(","))) if src_part.strip("[]") else []
            tgt_indices = list(map(int, tgt_part.strip("[]").split(","))) if tgt_part.strip("[]") else []
            parsed_alignments.append((src_indices, tgt_indices))

    print("\nBest Found:")
    print(f"del_percentile_frac: {best_del_percentile_frac:.3f} | Avg Cost: {best_avg_cost:.6f} | Zero-Cost Ratio: {best_zero_cost_ratio:.2%}")
    return parsed_alignments

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
    comet_scores = 0
    data = []
    # Write each window on a separate line
    for idx, window in enumerate(aggregated_windows):
        if window is None:
            zero_score_windows.append(idx)
        else:
            src, ref, mt = window
            if src and ref and mt:
                data.append({"src": src, "mt": mt, "ref": ref})
            else:
                zero_score_windows.append(idx)

    if data:
        model_outputs = model.predict(data, batch_size=8, gpus=1)
        comet_scores = model_outputs.scores  # list of float scores
    
    # Insert zero scores for windows that had missing scores
    for idx in zero_score_windows:
        comet_scores.insert(idx, 0.0)
    return comet_scores

def run_comet_qe_evaluation(aggregated_windows):
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)
    zero_score_windows = []
    comet_qe_scores = 0
    data = []

    # Write each window on a separate line
    for idx, window in enumerate(aggregated_windows):
        if window is None:
            zero_score_windows.append(idx)
        else:
            src, mt = window
            if src and mt:
                data.append({"src": src, "mt": mt})
            else:
                zero_score_windows.append(idx)
    if data:
        model_outputs = model.predict(data, batch_size=8, gpus=1)
        comet_qe_scores = model_outputs.scores
    
    # Insert zero scores for windows that had missing scores
    for idx in zero_score_windows:
        comet_qe_scores.insert(idx, 0.0)

    return comet_qe_scores

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
            "--tokenizer", "google/mt5-xl",
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
        src_merged = "\n".join([info["src_list"][i] for i in sorted_indices])
        tgt_merged = "\n".join([info["tgt_list"][i] for i in sorted_indices])
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
                "src_list": [],
                "seg_ids": []
            }
        merged[doc_id]["ref_list"].append(entry["tgt"])
        merged[doc_id]["src_list"].append(entry["src"])
        merged[doc_id]["seg_ids"].append(entry["seg_id"])
    
    for doc_id, info in merged.items():
        sorted_indices = sorted(range(len(info["seg_ids"])), key=lambda i: info["seg_ids"][i] if isinstance(info["seg_ids"][i], int) else int(info["seg_ids"][i].split('_')[0]))
        ref_merged = "\n".join([info["ref_list"][i] for i in sorted_indices])
        src_merged = "\n".join([info["src_list"][i] for i in sorted_indices])
        info["ref"] = ref_merged
        info["src"] = src_merged
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
            "ref": ref_merged.get(doc_id, {}).get("ref", ""),
            "src_list": ref_merged.get(doc_id, {}).get("src_list", ""),
            "ref_list": ref_merged.get(doc_id, {}).get("ref_list", "")
        })
    return combined


def aggregate_doc_id(doc_windows_list, window_key):
    """
    aggregate doc_id windows to dict： {doc_id: (start_index, window_count)}
    window_key: "ref_aligned" or "qe_aligned"
    """
    aggregated_lines = []
    mapping = {}
    current_index = 0
    for doc in doc_windows_list:
        windows = doc.get(window_key, [])
        mapping[doc["doc_id"]] = (current_index, len(windows))
        for win in windows:
            aggregated_lines.append(win)
        current_index += len(windows)
    return aggregated_lines, mapping

def prepare_doc_windows(doc):
    """
    Evaluate a single document (already merged)
    """
    src_sentences = doc["src_list"]
    ref_sentences = doc["ref_list"]
    tgt_text = doc["tgt"]

    # Sentence segmentation and preprocessing
    if SPACY == False:
        mt_sentences = segment_sentences_by_ersatz(tgt_text)
    else:
        mt_sentences = segment_sentences_by_spacy(tgt_text)

    # Generate overlap and embedding data
    src_overlap, src_embed = generate_overlap_and_embedding("\n".join(src_sentences))
    mt_overlap, mt_embed = generate_overlap_and_embedding("\n".join(mt_sentences))
    
    # Run vector alignment exploration
    src_mt_alignments = run_vecalign_explore("\n".join(src_sentences), "\n".join(mt_sentences), src_overlap, mt_overlap, src_embed, mt_embed)

    print("src_mt_alignments: ", src_mt_alignments)
    
    aligned_tuple = []
    aligned_qe_tuple = []
    for src_indices, mt_indices in src_mt_alignments:
        aligned_src = " ".join([src_sentences[i] for i in src_indices]) if src_indices else ""
        aligned_ref = " ".join([ref_sentences[i] for i in src_indices]) if src_indices else "" # src_indices == ref_indices
        aligned_mt= " ".join([mt_sentences[i] for i in mt_indices]) if mt_indices else ""
        aligned_tuple.append((aligned_src, aligned_ref, aligned_mt))
        aligned_qe_tuple.append((aligned_src, aligned_mt))

    return {
         "doc_id": doc["doc_id"],
         "sys_id": doc["sys_id"],
         "src": doc["src"],
         "tgt": doc["tgt"],
         "ref": doc["ref"],
         "ref_aligned": aligned_tuple,
         "qe_aligned": aligned_qe_tuple
    }


def prepare_doc_windows_with_retry(doc, retries=3):
    for attempt in range(1, retries+1):
        try:
            result = prepare_doc_windows(doc)
            return (doc["doc_id"], result)
        except Exception as e:
            print("\n################################################################################")
            print(f"prepare_doc_windows failed for doc {doc['doc_id']} on attempt {attempt}: {e}")
            print("################################################################################\n")
    return (doc["doc_id"], None)

def save_align_info(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            for i, (aligned_src, aligned_ref, aligned_mt) in enumerate(entry['ref_aligned']):
                record = {
                    "doc_id": entry["doc_id"],
                    "sys_id": entry["sys_id"],
                    "aligned_src": aligned_src,
                    "aligned_ref": aligned_ref,
                    "aligned_mt": aligned_mt
                }
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')

def init_config(task_lang):
    global mt_seg

    # Mapping from language name to the appropriate spaCy model name
    spacy_models = {
        "en": "en_core_web_sm",
        "ru": "ru_core_news_sm",
        "de": "de_core_news_sm",
        "zh": "zh_core_web_sm",
        "ja": "ja_ginza_electra",
        "es": "es_core_news_sm"
    }

    mt_seg = spacy.load(spacy_models[task_lang])

    print("Set SpaCy sentence segmentor")
# -----------------------------------------------------------------------------
# main function
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(description="Set TARGET_FILE, TARGET_COLUMN, and TASK_LANGUAGE")
    parser.add_argument("--system_file", type=str, required=True,
                        help="Path to the system JSONL file (e.g., GPT-4.jsonl)")
    parser.add_argument("--ref_file", type=str, required=True,
                        help="Path to the reference JSONL file (e.g., ref_A.jsonl)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output JSONL file")
    parser.add_argument("--pool_size", type=int, default=4,
                        help="Number of parallel processes")
    parser.add_argument("--segmenter", type=str, choices=["spacy", "ersatz"], required=True,
                        help="Sentence segmenter to use: 'spacy' or 'ersatz'")
    parser.add_argument("--task_lang", type=str, default="",
                        help="Target language (only used if segmenter is 'spacy')")
    args = parser.parse_args()

    # ------------------ Sentence Segmentor ------------------ # 
    global SPACY
    SPACY = args.segmenter == "spacy"
    if SPACY:
        if not args.task_lang:
            raise ValueError("When using --segmenter spacy, you must also specify --task_lang.")
        init_config(args.task_lang)

    # ------------------ If you want to evaluate the josnl files. ------------------ # 
    # Read system and reference files
    system_entries = read_jsonl(args.system_file)
    ref_entries = read_jsonl(args.ref_file)

    # Merge entries by doc_id separately
    system_merged = merge_system_entries(system_entries)
    ref_merged = merge_ref_entries(ref_entries)

    # Combine system and reference data to create a final list of documents for evaluation
    combined_docs = combine_system_ref(system_merged, ref_merged)

    # Evaluate each document in parallel
    with Pool(args.pool_size) as pool:
        results = list(tqdm(pool.imap(prepare_doc_windows_with_retry, combined_docs), total=len(combined_docs)))

    eval_doc_list = [res for (_, res) in results if res is not None]
    failed_doc_ids = [doc_id for (doc_id, res) in results if res is None]
    
    if len(failed_doc_ids) > 0:
        directory = os.path.dirname(args.system_file)
        failure_file = directory + "/failed_" + args.segmenter + "_" + args.system_file.split('.jsonl')[0].split('/')[-1] + ".jsonl"
        with open(failure_file, "w", encoding="utf-8") as f_fail:
            for entry in failed_doc_ids:
                f_fail.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Failed doc_id record: {failure_file}")

    if len(eval_doc_list) > 0:
        directory = os.path.dirname(args.system_file)
        aligned_file = directory + "/aligned_" + args.segmenter + "_" + args.system_file.split('.jsonl')[0].split('/')[-1] + ".jsonl"
        save_align_info(eval_doc_list, aligned_file)

        aggregated_comet_windows, comet_mapping = aggregate_doc_id(eval_doc_list, "ref_aligned")
        aggregated_qe_windows, qe_mapping = aggregate_doc_id(eval_doc_list, "qe_aligned")

        aggregated_comet_scores = run_comet_evaluation(aggregated_comet_windows)
        aggregated_comet_qe_scores = run_comet_qe_evaluation(aggregated_qe_windows)

        # follow same comet / comet_qe data, just different metrics
        aggregated_metricx_scores = run_metricx_evaluation(aggregated_comet_windows)
        aggregated_metricx_qe_scores = run_metricx_qe_evaluation(aggregated_qe_windows)

    final_results = []
    for doc in eval_doc_list:
        doc_id = doc["doc_id"]
        
        start, count = comet_mapping.get(doc_id, (0, 0))
        doc_comet_scores = aggregated_comet_scores[start:start+count]
        avg_comet = sum(doc_comet_scores) / len(doc_comet_scores) if doc_comet_scores else 0
        
        start_qe, count_qe = qe_mapping.get(doc_id, (0, 0))
        doc_comet_qe_scores = aggregated_comet_qe_scores[start_qe:start_qe+count_qe]
        avg_comet_qe = sum(doc_comet_qe_scores) / len(doc_comet_qe_scores) if doc_comet_qe_scores else 0
        
        # follow comet / comet_qe data, just different metrics

        start, count = comet_mapping.get(doc_id, (0, 0))
        doc_metricx_scores = aggregated_metricx_scores[start:start+count]
        avg_metricx = sum(doc_metricx_scores) / len(doc_metricx_scores) if doc_metricx_scores else 0
        
        start_qe, count_qe = qe_mapping.get(doc_id, (0, 0))
        doc_metricx_qe_scores = aggregated_metricx_qe_scores[start_qe:start_qe+count_qe]
        avg_metricx_qe = sum(doc_metricx_qe_scores) / len(doc_metricx_qe_scores) if doc_metricx_qe_scores else 0
        
        final_results.append({
            "doc_id": doc_id,
            "sys_id": doc["sys_id"],
            "src": doc["src"],
            "tgt": doc["tgt"], 
            "ref": doc["ref"],
            "comet": avg_comet,
            "comet-qe": avg_comet_qe,
            "metricx": avg_metricx,
            "metricx-qe": avg_metricx_qe
        })

    # Write evaluation results to a JSONL file
    with open(args.output_file, "w", encoding="utf-8") as f_out:
         for res in final_results:
              f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Evaluation completed at: {timestamp}. Results saved to: {args.output_file}")
