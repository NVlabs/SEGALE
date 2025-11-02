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
import datetime
import unicodedata
from tqdm import tqdm
from typing import Optional, List, Tuple
from transformers import AutoTokenizer, AutoModel
import pickle
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Global Variables
# -----------------------------------------------------------------------------
VECALIGN_DIR = "/home/shuoyangd/workspace/long-context-mt-eval/lcme-emnlp/vecalign"
LASER_DIR = "/home/shuoyangd/workspace/LASER"


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

def init_save_folder(system_file: str) -> str:
    """
    Initialize the save folder based on the system file's location and basename.
    """
    directory = os.path.dirname(system_file)
    sys_base = os.path.splitext(os.path.basename(system_file))[0]
    save_folder = os.path.join(directory, sys_base)
    os.makedirs(save_folder, exist_ok=True)
    return save_folder

# -----------------------------------------------------------------------------
# Sentence Segmentation Functions
# -----------------------------------------------------------------------------

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


def segment_sentences_by_spacy(text, src_or_tgt):
    segmented_sentences = []
    paragraphs = text.split('\n')
    for paragraph in paragraphs:
        if paragraph.strip():
            if src_or_tgt == "src":
                doc = src_seg(paragraph)
            elif src_or_tgt == "tgt":
                doc = tgt_seg(paragraph)
            for sent in doc.sents:
                segmented_sentences.append(sent.text.strip())
    return segmented_sentences

# -----------------------------------------------------------------------------
# Overlap and Embedding Functions
# -----------------------------------------------------------------------------


def compute_embedding_api(input_file: str, embed_file: str, model=None, tokenizer=None) -> bytes:
    """
    Compute embedding for an input file (e.g. overlaps file). If a transformer model is provided,
    use it; otherwise, use the embed.sh script. Ensures that embed.sh is called via its absolute path.
    """
    if model is not None and tokenizer is not None:        
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        if not lines:
            raise ValueError("Input file is empty.")

        tokens = tokenizer(
            lines,
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
            return_tensors="pt"
        )
        device = next(model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = model(**tokens)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]
            
            normalized_embeddings = F.normalize(embeddings, p=2)
            
            expected_dim = getattr(model.config, "hidden_size", 1024)
            if normalized_embeddings.shape[-1] != expected_dim:
                normalized_embeddings = normalized_embeddings[:, :expected_dim]
            normalized_embeddings = normalized_embeddings.cpu()
        
        emb_np = normalized_embeddings.numpy().astype(np.float32)
        emb_np.tofile(embed_file)
        print(f"Saved API embedding file (model) for {embed_file}")
    else:
        LASER_DIR = os.environ.get("LASER")
        if not LASER_DIR:
            raise ValueError("The LASER environment variable is not set.")
        embed_sh = os.path.join(LASER_DIR, "tasks", "embed", "embed.sh")
        cmd = f'"{embed_sh}" "{input_file}" "{embed_file}"'
        print(f"Executing command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        print(f"Saved API embedding file (laser) for {embed_file}")


def generate_overlap_and_embedding(text: str, tokenizer=None, model=None, max_size=10) -> tuple:
    """
    Generate overlap and embedding data from text using temporary files.
    The embedding computation is extracted to an external function.
    Args:
    text (str): Input text.
    doc_id (str): Document identifier (used for naming temporary files).
    save_folder (str): Ignored parameter for compatibility.
    model, tokenizer: Ignored parameters for compatibility.
    
    Returns:
    tuple: (overlap_content (str), embeddings_content (bytes))
    """
    with tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8", suffix=".txt") as txt_file:
        txt_file.write(text)
        txt_file.flush()
        txt_filename = txt_file.name
    
    # Define the paths for the overlaps and embedding files
    overlaps_file = txt_filename + ".overlaps"
    embed_file = txt_filename + ".emb"
    
    try:
        # Generate overlap data
        subprocess.run([os.path.join(VECALIGN_DIR, "overlap.py"), "-i", txt_filename, "-o", overlaps_file, "-n", str(max_size)], check=True)
        
        # Generate embedding data using the external function
        compute_embedding_api(overlaps_file, embed_file, model, tokenizer)
        
        # Read the contents
        with open(embed_file, "rb") as f:
            embeddings_content = f.read()
            print(len(embeddings_content))

        with open(overlaps_file, "r", encoding="utf-8") as f:
            overlap_content = f.read()
            
        return overlap_content, embeddings_content
    
    finally:
        # Clean up all temporary files
        for need_to_del_file in [txt_filename, overlaps_file, embed_file]:
            try:
                os.remove(need_to_del_file)
                print(f"Removed file: {need_to_del_file}")
            except Exception as e:
                print(f"Error removing {need_to_del_file}: {e}")


# -----------------------------------------------------------------------------
# Alternative Model Loading
# -----------------------------------------------------------------------------
def load_alternative_model(proc_device, alternative_model):
    """
    Load an alternative transformer model for API mode embedding.
    """
    tokenizer = AutoTokenizer.from_pretrained(alternative_model)
    model = AutoModel.from_pretrained(alternative_model, trust_remote_code=True)
    model = model.eval()
    model.to(proc_device)
    return tokenizer, model


# -----------------------------------------------------------------------------
# Alignment Functions
# -----------------------------------------------------------------------------


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


def find_best_alignment(
    all_results: List[dict], doc_id: str
) -> Tuple[List[Tuple[List[int], List[int]]], List[float], Tuple[float, float]]:
    """
    From collected vecalign outputs, find the best alignment by cost/penalty heuristics,
    and return zero-cost jumps and avg cost range for non-zero deletions.
    """
    best_result = None
    best_avg_cost = float("inf")

    # For extra info
    zero_cost_jumps = []
    nonzero_costs = []

    prev_zero = None

    for res in all_results:
        dpf = res["del_percentile_frac"]
        avg_cost = res["avg_cost"]
        zero_cost_ratio = res["zero_cost_ratio"]

        print(f"doc_id: {doc_id} | del_percentile_frac: {dpf:.3f} | Avg Cost: {avg_cost:.6f} | Zero-Cost Ratio: {zero_cost_ratio:.2%}")

        # track jumps in zero-cost ratio
        if prev_zero is not None and zero_cost_ratio != prev_zero:
            jump = zero_cost_ratio - prev_zero
            zero_cost_jumps.append(jump)
        prev_zero = zero_cost_ratio

        # collect non-zero deletions' avg cost
        if zero_cost_ratio > 0:
            nonzero_costs.append(avg_cost)

        # try to find a "perfect" one
        if zero_cost_ratio == 0 and avg_cost < best_avg_cost:
            best_result = res
            best_avg_cost = avg_cost

    # parse best alignment
    if best_result:
        print("\nBest Found:")
        print(f"doc_id: {doc_id} | del_percentile_frac: {best_result['del_percentile_frac']:.3f} | Avg Cost: {best_result['avg_cost']:.6f} | Zero-Cost Ratio: {best_result['zero_cost_ratio']:.2%}")
        alignments = parse_alignments(best_result["output_lines"])
    else:
        print("No valid alignment found.")
        alignments = []

    # compute cost range
    if nonzero_costs:
        avg_cost_range = (min(nonzero_costs), max(nonzero_costs))
    else:
        avg_cost_range = (None, None)

    return alignments, zero_cost_jumps, avg_cost_range


def parse_alignments(lines: List[str]) -> List[Tuple[List[int], List[int]]]:
    """
    Parse raw alignment lines into structured list of (src, tgt) index tuples.
    """
    parsed = []
    for line in lines:
        if line:
            src_part, tgt_part, _ = line.split(":")
            src_indices = list(map(int, src_part.strip("[]").split(","))) if src_part.strip("[]") else []
            tgt_indices = list(map(int, tgt_part.strip("[]").split(","))) if tgt_part.strip("[]") else []
            parsed.append((src_indices, tgt_indices))
    return parsed

def run_vecalign_explore(src_text: str, tgt_text: str, src_overlap: str, tgt_overlap: str,
                         src_embed: bytes, tgt_embed: bytes, doc_id: str, save_folder: str, max_size=8) -> List[Tuple[List[int], List[int]]]:
    """
    Explore vector alignment with different parameters and save all outputs to disk.
    Return the best alignment result based on predefined rules.
    """

    vecalign_folder = os.path.join(save_folder, "temp")
    os.makedirs(vecalign_folder, exist_ok=True)
    # Save inputs
    src_file_path = os.path.join(vecalign_folder, f"{doc_id}_src.txt")
    tgt_file_path = os.path.join(vecalign_folder, f"{doc_id}_tgt.txt")
    src_overlap_file_path = os.path.join(vecalign_folder, f"{doc_id}_src.overlaps")
    tgt_overlap_file_path = os.path.join(vecalign_folder, f"{doc_id}_tgt.overlaps")
    src_embed_file_path = os.path.join(vecalign_folder, f"{doc_id}_src.emb")
    tgt_embed_file_path = os.path.join(vecalign_folder, f"{doc_id}_tgt.emb")

    with open(src_file_path, "w+", encoding="utf-8") as f:
        f.write(src_text)
    with open(tgt_file_path, "w+", encoding="utf-8") as f:
        f.write(tgt_text)
    with open(src_overlap_file_path, "w+", encoding="utf-8") as f:
        f.write(src_overlap)
    with open(tgt_overlap_file_path, "w+", encoding="utf-8") as f:
        f.write(tgt_overlap)
    with open(src_embed_file_path, "wb") as f:
        f.write(src_embed)
    with open(tgt_embed_file_path, "wb") as f:
        f.write(tgt_embed)

    del_percentile_frac = 0.2
    step_size = 0.005
    all_results = []

    while del_percentile_frac > 0.01:
        result = subprocess.run(
            [
                os.path.join(VECALIGN_DIR, "vecalign.py"),
                "--alignment_max_size", str(max_size),
                "--del_percentile_frac", str(del_percentile_frac),
                "--src", src_file_path,
                "--tgt", tgt_file_path,
                "--src_embed", src_overlap_file_path, src_embed_file_path,
                "--tgt_embed", tgt_overlap_file_path, tgt_embed_file_path,
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        output_lines = result.stdout.strip().split("\n")
        avg_cost, zero_cost_ratio = compute_alignment_stats(output_lines)
        print(f"del_percentile_frac: {del_percentile_frac:.3f} | Avg Cost: {avg_cost:.6f} | Zero-Cost Ratio: {zero_cost_ratio:.2%}")


        all_results.append({
            "del_percentile_frac": del_percentile_frac,
            "avg_cost": avg_cost,
            "zero_cost_ratio": zero_cost_ratio,
            "output_lines": output_lines,
        })

        del_percentile_frac -= step_size

    # Save all results to JSON
    if VERBOSE >= 1:
        aps_folder = os.path.join(save_folder, f"{SPACY}_run_vecalign_explore")
        os.makedirs(aps_folder, exist_ok=True)
        json_path = os.path.join(aps_folder, f"{doc_id}_aps_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    best_alignments, zero_cost_jumps, avg_cost_range = find_best_alignment(all_results, doc_id)
    if not best_alignments:
        print("falling back to full alignment")
        src_len = len(src_text.split('\n'))
        tgt_len = len(tgt_text.split('\n'))
        best_alignments = [ (list(range(src_len)), list(range(tgt_len))) ]
    
    # clean up all the temp inputs/overlap/embed files
    for p in (
        src_file_path,
        tgt_file_path,
        src_overlap_file_path,
        tgt_overlap_file_path,
        src_embed_file_path,
        tgt_embed_file_path
    ):
        try:
            os.remove(p)
            print(f"Removed temporary file: {p}")
        except OSError as e:
            print(f"Failed to remove {p}: {e}")
        
    return best_alignments, zero_cost_jumps, avg_cost_range 

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

def merge_ref_entries(entries):
    """
    Merge reference JSONL (e.g., ref_A.jsonl) entries by doc_id.
    Concatenate the tgt fields for each doc_id to form the final reference text.
    """
    merged = {}
    sys_id = None
    for entry in entries:
        doc_id = entry["doc_id"]
        sys_id = entry["sys_id"]
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
        info["sys_id"] = sys_id
    return merged


def combine_system_ref(ref_merged):
    """
    Combine the merged system and reference data by doc_id.
    If a corresponding doc_id is not found in the reference, the ref field is set to an empty string.
    Returns a list where each element contains the system and reference texts.
    """
    combined = []
    for doc_id, ref_info in ref_merged.items():
        combined.append({
            "doc_id": doc_id,
            "sys_id": ref_info["sys_id"],
            "src": ref_info["src"],
            "ref": ref_info["ref"],
            "src_list": ref_merged.get(doc_id, {}).get("src_list", ""),
            "ref_list": ref_merged.get(doc_id, {}).get("ref_list", "")
        })
    return combined


def prepare_doc_windows(doc, save_folder, tokenizer, model, max_size):
    """
    Evaluate a single document (already merged)
    """
    ref_text = doc["ref"]
    doc_id = doc["doc_id"]

    if TWO_SIDE:
        src_text = doc["src"]
        # Sentence segmentation and preprocessing
        if SPACY != "spacy":
            src_sentences = segment_sentences_by_ersatz(src_text)
            ref_sentences = segment_sentences_by_ersatz(ref_text)
        else:
            src_sentences = segment_sentences_by_spacy(src_text, "src")
            ref_sentences = segment_sentences_by_spacy(ref_text, "tgt")
    else:
        src_sentences = doc["src_list"]
        
        indices_to_remove = [index for index, item in enumerate(src_sentences) if item == ""]
        if indices_to_remove:
            print(f"WARNING: Empty string detected in source, doc_id = {doc_id}. dropped.")
        src_sentences = [item for index, item in enumerate(src_sentences) if index not in indices_to_remove]

        if SPACY != "spacy":
            ref_sentences = segment_sentences_by_ersatz(ref_text)
        else:
            ref_sentences = segment_sentences_by_spacy(ref_text, "tgt")

    good_max_size = None
    for max_size in range(4, max_size+1, 2):
        print(f"Currently trying to set the overlap parameter to align with {max_size}")
        # Generate overlap and embedding data
        src_overlap, src_embed = generate_overlap_and_embedding("\n".join(src_sentences), tokenizer, model, max_size)
        ref_overlap, ref_embed = generate_overlap_and_embedding("\n".join(ref_sentences), tokenizer, model, max_size)
        # Run vector alignment exploration
        src_ref_alignments, jump_list, cost_range = run_vecalign_explore("\n".join(src_sentences), "\n".join(ref_sentences), src_overlap, ref_overlap, src_embed, ref_embed, doc_id, save_folder, max_size)
        if len(src_ref_alignments) != 0:
            good_max_size = max_size
            print(f"The optimal overlap has been found: overlap = {good_max_size}.")
            break
    
    if len(src_ref_alignments) == 0:
        print(f"When the optimal overlap could not be identified, the default maximum value of {max_size} was used.")
        good_max_size = max_size

    print("initial alignment results: ")
    print("src_ref_alignments: ", src_ref_alignments)
    
    # For reference-free evaluation: get non-adjusted alignments
    aligned_src_ref = []
    for src_indices, ref_indices in src_ref_alignments:
        aligned_src = " ".join([src_sentences[i] for i in src_indices]) if src_indices else ""
        aligned_ref = " ".join([ref_sentences[i] for i in ref_indices]) if ref_indices else ""
        aligned_src_ref.append((aligned_src, aligned_ref))

    result_dict = {
         "doc_id": doc["doc_id"],
         "sys_id": doc["sys_id"],
         "aligned_src_ref": aligned_src_ref,
         "jump_list": jump_list,
         "cost_range": cost_range,
         "good_max_size": good_max_size
    }

    if VERBOSE >= 2:
        individual_folder = os.path.join(save_folder, f"{SPACY}_individual_alignments")
        os.makedirs(individual_folder, exist_ok=True)
        individual_file = os.path.join(individual_folder, f"{doc_id}.json")
        with open(individual_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        print(f"Saved individual alignment for doc_id {doc_id} at {individual_file}")

    return result_dict

def save_align_info(results, output_path):
    """
    Save alignment results to a JSONL file.

    Args:
        results (list): List of dictionaries containing alignment results.
        output_path (str): Path to save the JSONL file.
    """
    seg_id = 1
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            doc_id = result.get("doc_id", "unknown")
            sys_id = result.get("sys_id", "unknown")
            aligned_src_ref = result.get("aligned_src_ref", [])
            for aligned_src, aligned_ref in aligned_src_ref:
                record = {
                    "src": aligned_src,
                    "tgt": aligned_ref,
                    "sys_id": sys_id,
                    "doc_id": doc_id,
                    "seg_id": seg_id
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                seg_id += 1

def init_config(src_lang, task_lang):
    global tgt_seg, src_seg

    # Mapping from language name to the appropriate spaCy model name
    spacy_models = {
        "en": "en_core_web_sm",
        "ru": "ru_core_news_sm",
        "de": "de_core_news_sm",
        "zh": "zh_core_web_sm",
        "ja": "ja_ginza_electra",
        "es": "es_core_news_sm"
    }

    tgt_seg = spacy.load(spacy_models[task_lang])
    src_seg = spacy.load(spacy_models[src_lang])
    print("Set SpaCy sentence segmentor")


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(description="Set TARGET_FILE, TARGET_COLUMN, and TASK_LANGUAGE")
    parser.add_argument("--ref_file", type=str, required=True,
                        help="Path to the reference JSONL file (e.g., ref_A.jsonl)")
    parser.add_argument("--segmenter", type=str, choices=["spacy", "ersatz"], required=True,
                        help="Sentence segmenter to use: 'spacy' or 'ersatz'")
    parser.add_argument("--two_side", action="store_true",
                    help="Enable two-side alignment mode (default: False)")
    parser.add_argument("--src_lang", type=str, default="",
                        help="Sourcr language (only used if segmenter is 'spacy')")
    parser.add_argument("--task_lang", type=str, default="",
                        help="Target language (only used if segmenter is 'spacy')")
    parser.add_argument("--overlap", type=int, default=10,
                        help="Searches for alignments up to size N-M, where N+M <= this value. Note that the the embeddings must support the requested number of overlaps")
    parser.add_argument("--proc_device", type=str, default="cpu",
                        help="Device to process alternative embedding (e.g., 'cpu' or 'cuda')")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="increase verbosity: -v save all_results json; -vv save individual alignments")
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    global TWO_SIDE
    if args.two_side:
        print("Two-side alignment is enabled.")
        TWO_SIDE = True
    else:
        print("Using one-side alignment.")
        TWO_SIDE = False


    dir_path = os.path.dirname(args.ref_file)
    base_name_without_ext = os.path.splitext(os.path.basename(args.ref_file))[0]

    if TWO_SIDE:
        SAVE_FOLDER = init_save_folder(os.path.join(dir_path, args.segmenter + "_" + base_name_without_ext))
    else:
        SAVE_FOLDER = init_save_folder(os.path.join(dir_path, base_name_without_ext))
    print(f"Save folder: {SAVE_FOLDER}")

    # ------------------ Sentence Segmentor ------------------ # 
    global SPACY
    SPACY = args.segmenter
    if SPACY == "spacy":
        if not args.task_lang:
            raise ValueError("When using --segmenter spacy, you must also specify --task_lang.")
        init_config(args.src_lang, args.task_lang)
    
    ref_entries = read_jsonl(args.ref_file)
    ref_merged = merge_ref_entries(ref_entries)
    combined_docs = combine_system_ref(ref_merged)

    # Uncomment to use the transformer-based embedding; otherwise, fallback to embed.sh
    # tokenizer, model = load_alternative_model(args.proc_device, 'BAAI/bge-m3')
    # tokenizer, model = load_alternative_model(args.proc_device, 'intfloat/multilingual-e5-large')
    # Alternatively: 
    tokenizer, model = None, None

    results = []
    for doc in tqdm(combined_docs):
        result = prepare_doc_windows(doc, SAVE_FOLDER, tokenizer, model, args.overlap)
        results.append(result)
    directory = os.path.dirname(args.ref_file)
    if TWO_SIDE:
        aligned_file = directory + "/" + args.segmenter + "_" + args.ref_file.split('.jsonl')[0].split('/')[-1] + ".jsonl"
    else:
        aligned_file = directory + "/" + args.segmenter + "_p_" + args.ref_file.split('.jsonl')[0].split('/')[-1] + ".jsonl"
    save_align_info(results, aligned_file)


    # --- New: aggregate jump and cost stats across all docs ---
    all_jumps = [j for r in results for j in r.get("jump_list", [])]
    if all_jumps:
        p10 = np.percentile(all_jumps, 20)
        p90 = np.percentile(all_jumps, 80)
        trimmed = [j for j in all_jumps if p10 <= j <= p90]
        if trimmed:
            avg_jump = sum(trimmed) / len(trimmed)
            min_jump = min(trimmed)
            max_jump = max(trimmed)
        else:
            avg_jump = min_jump = max_jump = None
    else:
        avg_jump = min_jump = max_jump = None

    all_cost_vals = []
    for r in results:
        cr = r.get("cost_range", (None, None))
        if cr[0] is not None: all_cost_vals.append(cr[0])
        if cr[1] is not None: all_cost_vals.append(cr[1])
    if all_cost_vals:
        cost_union = (min(all_cost_vals), max(all_cost_vals))
    else:
        cost_union = (None, None)

    summary = {
        "jump_stats": {"avg": avg_jump, "min": min_jump, "max": max_jump},
        "cost_range_union": {"min": cost_union[0], "max": cost_union[1]},
        "conservative_overlap_size": max([r.get("good_max_size", 10) for r in results])
    }

    summary_path = os.path.join(SAVE_FOLDER, f"{SPACY}_alignment_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("1-0/0-1 ratio jump stats:", summary["jump_stats"])
    print("alignment cost range:", summary["cost_range_union"])
    print("conservative overlap size:", summary["conservative_overlap_size"])
