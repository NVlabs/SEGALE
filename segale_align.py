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
import json
import spacy
import torch
import random
import argparse
import numpy as np
import tempfile
import subprocess
import datetime

from tqdm import tqdm
from typing import Optional, List, Tuple
from transformers import AutoTokenizer, AutoModel
from vecalign.dp_utils import yield_overlaps

import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Global Variables
# -----------------------------------------------------------------------------
LASER_DIR = "/opt/LASER"


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """
    Set the global random seed for reproducibility.
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
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w+", encoding="utf-8", suffix=".txt"
    ) as temp_in:
        temp_in.write(text)
        temp_in.flush()
        input_filename = temp_in.name
    output_filename = input_filename + ".segmented"
    subprocess.run(
        ["ersatz", "--input", input_filename, "--output", output_filename], check=True
    )
    with open(output_filename, "r", encoding="utf-8") as f:
        segmented_text = f.read()
    os.remove(input_filename)
    os.remove(output_filename)
    sentences = [line.strip() for line in segmented_text.splitlines() if line.strip()]
    return sentences


def segment_sentences_by_spacy(text: str) -> list:
    """
    Segment sentences using spaCy.
    """
    segmented_sentences = []
    paragraphs = text.split("\n")
    for paragraph in paragraphs:
        if paragraph.strip():
            doc = mt_seg(paragraph)
            for sent in doc.sents:
                segmented_sentences.append(sent.text.strip())
    return segmented_sentences


# -----------------------------------------------------------------------------
# Overlap and Embedding Functions
# -----------------------------------------------------------------------------

# def compute_embedding_api(input_file: str, output_file: str):
#     """
#     Compute embedding for an input file using the LASER embed.sh script.
#     """
#     # Use the embed.sh script as in the original code
#     subprocess.run(" ".join(["$LASER/tasks/embed/embed.sh", input_file, output_file]),
#                   shell=True, check=True)


def compute_embedding_api(overlaps: list[str], model=None, tokenizer=None) -> bytes:
    """
    Compute embedding for an input file (e.g. overlaps file). If a transformer model is provided,
    use it; otherwise, use the embed.sh script. Ensures that embed.sh is called via its absolute path.
    """

    if tokenizer is not None:
        tokens = tokenizer(
            overlaps,
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
            return_tensors="pt",
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

        return [emb.numpy().astype(np.float32) for emb in normalized_embeddings]

        # emb_np = normalized_embeddings.numpy().astype(np.float32)
        # emb_np.tofile(embed_file)
        # print(f"Saved API embedding file (model) for {embed_file}")
    else:
        return [model.encode_sentences([overlap])[0] for overlap in overlaps]


def generate_overlap_and_embedding(
    texts: list[str], model=None, tokenizer=None, max_size=10
) -> tuple:
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

    # # Generate overlap data
    # overlaps = []
    # for text in texts:
    #     lines = text.split('\n')
    #     output = set()
    #     for out_line in yield_overlaps(lines, max_size):
    #         output.add(out_line)
    #     output = list(output)
    #     output.sort()
    #     overlaps.append(output)

    # Generate embedding data using the external function
    overlaps_content = sorted(
        list(set([overlap for overlap in yield_overlaps(texts, max_size)]))
    )
    embeddings = compute_embedding_api(overlaps_content, model, tokenizer)
    # overlaps_content = ['\n'.join(overlap) for overlap in overlaps]

    return overlaps_content, embeddings


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
    Compute the average alignment cost (ignoring zero-cost alignments)
    and the zero-cost ratio.
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
    zero_cost_ratio = (
        zero_cost_count / len(alignment_results) if alignment_results else 0.0
    )
    return avg_cost, zero_cost_ratio


def find_best_alignment(
    all_results: List[dict], doc_id: str
) -> List[Tuple[List[int], List[int]]]:
    """
    Based on heuristic rules, find the best alignment output from collected results.
    """
    best_result = None
    best_avg_cost = float("inf")
    prev_zero_cost_ratio = None
    prev_avg_cost = None
    fallback_result = None

    for res in all_results:
        dpf = res["del_percentile_frac"]
        avg_cost = res["avg_cost"]
        zero_cost_ratio = res["zero_cost_ratio"]

        # Save the result with dpf = 0.3 for fallback
        if abs(dpf - 0.2) < 1e-6:
            fallback_result = res

        # Check stopping criteria
        if (
            prev_zero_cost_ratio is not None
            and prev_zero_cost_ratio != 0
            and (zero_cost_ratio / prev_zero_cost_ratio) > 1.5
        ):
            print(f"Stopping exploration at {dpf:.3f}")
            break
        elif prev_zero_cost_ratio is not None and (
            (zero_cost_ratio - prev_zero_cost_ratio) > STOP_JUMP
            or avg_cost > prev_avg_cost
            or avg_cost < COST_MIN
            or zero_cost_ratio > 0.7
        ):
            print(f"Stopping exploration at {dpf:.3f}")
            break
        else:  # Keep best one so far
            if avg_cost < best_avg_cost:
                best_result = res
                best_avg_cost = avg_cost

        prev_zero_cost_ratio = zero_cost_ratio
        prev_avg_cost = avg_cost

    if best_result:
        print("\nBest Found:")
        print(
            f"doc_id: {doc_id} | del_percentile_frac: {best_result['del_percentile_frac']:.3f} | Avg Cost: {best_result['avg_cost']:.6f} | Zero-Cost Ratio: {best_result['zero_cost_ratio']:.2%}"
        )
        return parse_alignments(best_result["output_lines"])
    elif fallback_result:
        print("\nFallback to del_percentile_frac = 0.3")
        print(
            f"doc_id: {doc_id} | Avg Cost: {fallback_result['avg_cost']:.6f} | Zero-Cost Ratio: {fallback_result['zero_cost_ratio']:.2%}"
        )
        return parse_alignments(fallback_result["output_lines"])
    else:
        print("No valid alignment found.")
        return []


def parse_alignments(lines: List[str]) -> List[Tuple[List[int], List[int]]]:
    parsed = []
    for line in lines:
        if line:
            src_part, tgt_part, _ = line.split(":")
            src_indices = (
                list(map(int, src_part.strip("[]").split(",")))
                if src_part.strip("[]")
                else []
            )
            tgt_indices = (
                list(map(int, tgt_part.strip("[]").split(",")))
                if tgt_part.strip("[]")
                else []
            )
            parsed.append((src_indices, tgt_indices))
    return parsed


def run_vecalign_explore(
    src_text: str,
    tgt_text: str,
    src_overlap: str,
    tgt_overlap: str,
    src_embed: list,
    tgt_embed: list,
    doc_id: str,
    save_folder: str,
    max_size=10,
) -> List[Tuple[List[int], List[int]]]:
    """
    Run vecalign at different del_percentile_frac settings and store all results.
    Return the best alignment based on computed stats.

    For debugging:
    with open("/path/to/{doc_id}_all_results.json", encoding="utf-8") as f:
        all_results = json.load(f)
    best = find_best_alignment(all_results, doc_id="my_doc_id")
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
        f.write(np.concatenate(src_embed, axis=0).tobytes())
    with open(tgt_embed_file_path, "wb") as f:
        f.write(np.concatenate(tgt_embed, axis=0).tobytes())

    del_percentile_frac = 0.2
    step_size = 0.005
    all_results = []

    while del_percentile_frac > 0.01:
        result = subprocess.run(
            [
                "vecalign",
                "--alignment_max_size",
                str(max_size),
                "--del_percentile_frac",
                str(del_percentile_frac),
                "--src",
                src_file_path,
                "--tgt",
                tgt_file_path,
                "--src_embed",
                src_overlap_file_path,
                src_embed_file_path,
                "--tgt_embed",
                tgt_overlap_file_path,
                tgt_embed_file_path,
            ],
            stdout=subprocess.PIPE,
            text=True,
        )

        output_lines = result.stdout.strip().split("\n")
        avg_cost, zero_cost_ratio = compute_alignment_stats(output_lines)

        print(
            f"doc_id: {doc_id} | del_percentile_frac: {del_percentile_frac:.3f} | Avg Cost: {avg_cost:.6f} | Zero-Cost Ratio: {zero_cost_ratio:.2%}"
        )

        all_results.append(
            {
                "del_percentile_frac": del_percentile_frac,
                "avg_cost": avg_cost,
                "zero_cost_ratio": zero_cost_ratio,
                "output_lines": output_lines,
            }
        )

        del_percentile_frac -= step_size

    if VERBOSE >= 1:
        # Save all results to JSON
        aps_folder = os.path.join(save_folder, f"{SPACY}_run_vecalign_explore")
        os.makedirs(aps_folder, exist_ok=True)
        json_path = os.path.join(aps_folder, f"{doc_id}_aps_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    # clean up all the temp inputs/overlap/embed files
    for p in (
        src_file_path,
        tgt_file_path,
        src_overlap_file_path,
        tgt_overlap_file_path,
        src_embed_file_path,
        tgt_embed_file_path,
    ):
        try:
            os.remove(p)
            print(f"Removed temporary file: {p}")
        except OSError as e:
            print(f"Failed to remove {p}: {e}")

    best_alignments = find_best_alignment(all_results, doc_id)
    return best_alignments


# -----------------------------------------------------------------------------
# Alternative File Merging Functions
# -----------------------------------------------------------------------------

def load_alignment_summary(ref_save_folder):
    """
    Load the alignment_summary.json and return jump stats and cost range.

    Returns:
        dict: {
            'avg_jump': float or None,
            'min_jump': float or None,
            'max_jump': float or None,
            'cost_min': float or None,
            'cost_max': float or None,
            'overlap' : int or None,
        }
    """
    summary_path = os.path.join(ref_save_folder, f"{SPACY}_alignment_summary.json")
    if not os.path.exists(summary_path) or summary_path == None:
        return {
            "avg_jump": 0.15,
            "min_jump": 0.15,
            "max_jump": 0.15,
            "cost_min": 0.30,
            "cost_max": 0.30,
            "overlap": 10,
        }
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    jump = summary.get("jump_stats", {})
    cost = summary.get("cost_range_union", {})
    overlap = summary.get("conservative_overlap_size", {})
    return {
        "avg_jump": jump.get("avg"),
        "min_jump": jump.get("min"),
        "max_jump": jump.get("max"),
        "cost_min": cost.get("min"),
        "cost_max": cost.get("max"),
        "overlap": overlap,
    }


def read_jsonl(file_path):
    """
    Read a JSONL file and return a list of JSON objects.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def merge_system_entries(entries):
    """
    Merge multiple system JSONL entries by doc_id.
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
                "seg_ids": [],
            }
        merged[doc_id]["src_list"].append(entry["src"])
        merged[doc_id]["tgt_list"].append(entry["tgt"])
        merged[doc_id]["seg_ids"].append(entry["seg_id"])

    for doc_id, info in merged.items():
        sorted_indices = sorted(
            range(len(info["seg_ids"])), key=lambda i: info["seg_ids"][i]
        )
        info["src"] = "\n".join([info["src_list"][i] for i in sorted_indices])
        info["tgt"] = "\n".join([info["tgt_list"][i] for i in sorted_indices])
    return merged


def merge_ref_entries(entries):
    """
    Merge reference JSONL entries by doc_id.
    """
    merged = {}
    for entry in entries:
        doc_id = entry["doc_id"]
        if doc_id not in merged:
            merged[doc_id] = {
                "doc_id": doc_id,
                "ref_list": [],
                "src_list": [],
                "seg_ids": [],
            }
        merged[doc_id]["ref_list"].append(entry["tgt"])
        merged[doc_id]["src_list"].append(entry["src"])
        merged[doc_id]["seg_ids"].append(entry["seg_id"])

    for doc_id, info in merged.items():
        sorted_indices = sorted(
            range(len(info["seg_ids"])),
            key=lambda i: (
                info["seg_ids"][i]
                if isinstance(info["seg_ids"][i], int)
                else int(info["seg_ids"][i].split("_")[0])
            ),
        )
        info["ref"] = "\n".join([info["ref_list"][i] for i in sorted_indices])
        info["src"] = "\n".join([info["src_list"][i] for i in sorted_indices])
    return merged


def combine_system_ref(system_merged, ref_merged):
    """
    Combine the merged system and reference data by doc_id.
    """
    combined = []
    for doc_id, sys_info in system_merged.items():
        combined.append(
            {
                "doc_id": doc_id,
                "sys_id": sys_info["sys_id"],
                "src": sys_info["src"],
                "tgt": sys_info["tgt"],
                "ref": ref_merged.get(doc_id, {}).get("ref", ""),
                "src_list": ref_merged.get(doc_id, {}).get("src_list", ""),
                "ref_list": ref_merged.get(doc_id, {}).get("ref_list", ""),
            }
        )
    return combined


def aggregate_doc_id(doc_windows_list, window_key):
    """
    Aggregate doc_id windows to a dictionary.
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


def clean_lists(primary_list, secondary_list, doc_id):
    if len(primary_list) != len(secondary_list):
        raise ValueError("Both lists must have the same length.")
    indices_to_remove = [index for index, item in enumerate(primary_list) if item == ""]
    if indices_to_remove:
        print(f"WARNING: Empty string detected in source, doc_id = {doc_id}. dropped.")
    # Remove items from both lists based on identified indices
    primary_list = [
        item
        for index, item in enumerate(primary_list)
        if index not in indices_to_remove
    ]
    secondary_list = [
        item
        for index, item in enumerate(secondary_list)
        if index not in indices_to_remove
    ]
    return primary_list, secondary_list


# -----------------------------------------------------------------------------
# Document Window Preparation Functions
# -----------------------------------------------------------------------------

def prepare_doc_windows(doc, save_folder, tokenizer=None, model=None, max_size=8):
    """
    Process a single merged document to prepare alignment windows.
    """
    doc_id = doc["doc_id"]
    src_sentences = doc["src_list"]
    ref_sentences = doc["ref_list"]
    src_sentences, ref_sentences = clean_lists(src_sentences, ref_sentences, doc_id)
    tgt_text = doc["tgt"]

    if SPACY != "spacy":
        mt_sentences = segment_sentences_by_ersatz(tgt_text)
    else:
        mt_sentences = segment_sentences_by_spacy(tgt_text)

    src_overlap, src_embed = generate_overlap_and_embedding(
        src_sentences, model, tokenizer, max_size
    )
    mt_overlap, mt_embed = generate_overlap_and_embedding(
        mt_sentences, model, tokenizer, max_size
    )

    src_mt_alignments = run_vecalign_explore(
        "\n".join(src_sentences),
        "\n".join(mt_sentences),
        "\n".join(src_overlap),
        "\n".join(mt_overlap),
        src_embed,
        mt_embed,
        doc_id,
        save_folder,
        max_size,
    )

    print("src_mt_alignments: ", src_mt_alignments)

    aligned_tuple = []
    aligned_qe_tuple = []
    for src_indices, mt_indices in src_mt_alignments:
        aligned_src = (
            " ".join([src_sentences[i] for i in src_indices]) if src_indices else ""
        )
        aligned_ref = (
            " ".join([ref_sentences[i] for i in src_indices]) if src_indices else ""
        )
        aligned_mt = (
            " ".join([mt_sentences[i] for i in mt_indices]) if mt_indices else ""
        )
        aligned_tuple.append((aligned_src, aligned_ref, aligned_mt))
        aligned_qe_tuple.append((aligned_src, aligned_mt))

    result_dict = {
        "doc_id": doc["doc_id"],
        "sys_id": doc["sys_id"],
        "src": doc["src"],
        "tgt": doc["tgt"],
        "ref": doc["ref"],
        "ref_aligned": aligned_tuple,
        "qe_aligned": aligned_qe_tuple,
    }

    if VERBOSE >= 2:
        individual_folder = os.path.join(save_folder, f"{SPACY}_individual_alignments")
        os.makedirs(individual_folder, exist_ok=True)
        individual_file = os.path.join(individual_folder, f"{doc_id}.json")
        with open(individual_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        print(f"Saved individual alignment for doc_id {doc_id} at {individual_file}")

    return result_dict


def save_align_info(data, filename):
    """
    Save alignment information into a JSONL file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for entry in data:
            for i, (aligned_src, aligned_ref, aligned_mt) in enumerate(
                entry["ref_aligned"]
            ):
                record = {
                    "doc_id": entry["doc_id"],
                    "sys_id": entry["sys_id"],
                    "src": aligned_src,
                    "ref": aligned_ref,
                    "tgt": aligned_mt,
                    "seg_id": i + 1,
                }
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")


def init_config(task_lang):
    """
    Initialize the spaCy sentence segmenter based on the target language.
    """
    global mt_seg
    spacy_models = {
        "en": "en_core_web_sm",
        "ru": "ru_core_news_sm",
        "de": "de_core_news_sm",
        "zh": "zh_core_web_sm",
        "ja": "ja_ginza_electra",
        "es": "es_core_news_sm",
    }
    mt_seg = spacy.load(spacy_models[task_lang])
    print("Set spaCy sentence segmenter")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    set_seed(42)
    parser = argparse.ArgumentParser(
        description="Set TARGET_FILE, TARGET_COLUMN, and TASK_LANGUAGE"
    )
    parser.add_argument(
        "--system_file",
        type=str,
        required=True,
        help="Path to the system JSONL file (e.g., GPT-4.jsonl)",
    )
    parser.add_argument(
        "--ref_file",
        type=str,
        required=True,
        help="Path to the reference JSONL file (e.g., ref_A.jsonl)",
    )
    parser.add_argument(
        "--segmenter",
        type=str,
        choices=["spacy", "ersatz"],
        required=True,
        help="Sentence segmenter to use: 'spacy' or 'ersatz'",
    )
    parser.add_argument(
        "--task_lang",
        type=str,
        default="",
        help="Target language (only used if segmenter is 'spacy')",
    )
    parser.add_argument(
        "--proc_device",
        type=str,
        default="cpu",
        help="Device to process alternative embedding (e.g., 'cpu' or 'cuda')",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase verbosity: -v save all_results json; -vv save individual alignments",
    )
    parser.add_argument(
        "--max_size", type=int, default=8, help="Maximum size for alignment windows"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="Embedding model to use -- can be a path to a model folder or a huggingface model name. If not provided, use LASER",
    )
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    SAVE_FOLDER = os.path.abspath(init_save_folder(args.system_file))
    print(f"Save folder: {SAVE_FOLDER}")

    global SPACY, STOP_JUMP, COST_MAX, COST_MIN
    SPACY = args.segmenter
    if SPACY == "spacy":
        if not args.task_lang:
            raise ValueError(
                "When using --segmenter spacy, you must also specify --task_lang."
            )
        init_config(args.task_lang)

    ref_dir_path = os.path.dirname(args.ref_file)
    ref_name_without_ext = os.path.splitext(os.path.basename(args.ref_file))[0]
    align_paras = load_alignment_summary(
        os.path.join(ref_dir_path, ref_name_without_ext)
    )
    # Uncomment to use the pre-defined parameters
    # align_paras = load_alignment_summary(None)

    print("align_paras: ", align_paras)
    STOP_JUMP = align_paras["min_jump"]
    COST_MAX = align_paras["cost_max"]
    COST_MIN = align_paras["cost_min"]
    MAX_OVERLAP = align_paras["overlap"]

    system_entries = read_jsonl(args.system_file)
    ref_entries = read_jsonl(args.ref_file)

    system_merged = merge_system_entries(system_entries)
    ref_merged = merge_ref_entries(ref_entries)

    combined_docs = combine_system_ref(system_merged, ref_merged)

    if args.embedding_model is None:
        try:
            from laser_encoders import LaserEncoderPipeline
        except ImportError as e:
            raise ImportError(
                "laser_encoders is required when --embedding_model is not specified. Install it with `pip install laser_encoders==0.0.2`."
            ) from e
        tokenizer, model = (
            None,
            LaserEncoderPipeline(model_dir=LASER_DIR, laser="laser2"),
        )
    else:
        tokenizer, model = load_alternative_model(
            args.proc_device, args.embedding_model
        )

    sequential_results = []
    failed_doc_ids = []

    for doc in tqdm(combined_docs, desc="Processing documents"):
        doc_id = doc["doc_id"]
        result = prepare_doc_windows(doc, SAVE_FOLDER, tokenizer, model, args.max_size)

        if result is not None:
            sequential_results.append(result)
        else:
            failed_doc_ids.append(doc_id)

    if failed_doc_ids:
        failure_file = os.path.join(
            SAVE_FOLDER,
            f"failed_{args.segmenter}_{os.path.splitext(os.path.basename(args.system_file))[0]}.jsonl",
        )
        with open(failure_file, "w", encoding="utf-8") as f_fail:
            for doc_id in failed_doc_ids:
                f_fail.write(json.dumps(doc_id, ensure_ascii=False) + "\n")
        print(f"Failed doc_id record: {failure_file}")

    if sequential_results:
        aligned_file = os.path.join(
            SAVE_FOLDER,
            f"aligned_{args.segmenter}_{os.path.splitext(os.path.basename(args.system_file))[0]}.jsonl",
        )
        save_align_info(sequential_results, aligned_file)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Alignment completed at: {timestamp}.")


if __name__ == "__main__":
    main()
