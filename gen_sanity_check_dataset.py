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

"""
Script: gen_all_sanity_check_dataset_combined.py

Description:
  This script combines the functionality of performing merge or drop operations on a machine translation dataset
  (using GPT-4 API and BLEURT checks) with batch processing over multiple folders.
  
  For each of the following folders:
      - data/wmt24/json_output_en_de
      - data/wmt24/json_output_en_es
      - data/wmt24/json_output_ja_zh
  
  Only "ref_A.jsonl" is processed (ref_B.jsonl is excluded). The script performs four operations:
      - merge
      - drop_src
      - drop_tgt
  
  After processing ref_A.jsonl for a given operation, the script updates the other JSONL files in the same folder
  based on the new ref_A (note that merge may change the number of segments).
  
  The outputs for each operation are stored in a subfolder (named after the operation) within the original folder.
  
Usage:
  python gen_all_sanity_check_dataset.py
"""

import os
import json
import random
import math
import argparse
import openai
from copy import deepcopy
import tiktoken
from bleurt import score as bleurt_score_module

# -------------------------
# Configurable Parameters
# -------------------------
SAMPLE_RATE = 0.1             # Proportion of eligible segments to process
BLEURT_THRESHOLD = None         # Minimum BLEURT similarity score required
TOKEN_LENGTH_THRESHOLD = 64   # Minimum token count for split operations
MAX_ATTEMPTS = 3              # Maximum rewriting attempts per candidate

# Global BLEURT scorer (initialized with checkpoint "BLEURT-20")
bleurt_scorer = bleurt_score_module.BleurtScorer("../BLEURT-20")
# Preinitialize tiktoken encoder for GPT-4 ("cl100k_base" encoding)
encoder = tiktoken.get_encoding("cl100k_base")
# Set your OpenAI API key here (or set the environment variable OPENAI_API_KEY)
openai.api_key = ""

# Global variables for rewriting process logging.
LOG_REWRITING = False
rewriting_logs = []

# -------------------------
# Helper Functions
# -------------------------
def load_jsonl(file_path):
    """
    Load a JSONL file and return a list of JSON objects.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def write_jsonl(file_path, data):
    """
    Write a list of JSON objects to a JSONL file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

def count_tokens(text):
    """
    Count tokens in a text string using tiktoken (GPT-4's "cl100k_base" encoding).
    """
    return len(encoder.encode(text))

def call_gpt4_api(prompt):
    """
    Call the GPT-4 API using OpenAI's ChatCompletion endpoint.
    Note: the max_tokens parameter is omitted.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
    )
    return response['choices'][0]['message']['content'].strip()

def bleurt_score(original, rewritten):
    """
    Calculate the BLEURT similarity score between original and rewritten texts.
    """
    scores = bleurt_scorer.score(references=[original], candidates=[rewritten])
    return scores[0]

def group_by_doc(data):
    """
    Group the list of segments by 'doc_id'.
    Returns a dictionary mapping doc_id to a list of segments.
    Assumes the data are in their original order.
    """
    docs = {}
    for seg in data:
        doc_id = seg["doc_id"]
        docs.setdefault(doc_id, []).append(seg)
    return docs

# -------------------------
# Operation Functions
# -------------------------
def process_merge(docs, sample_rate, bleurt_threshold):
    """
    Process merge operations on eligible segments.
    For each document:
      - Eligible segments are those that are not the first and not the last segment.
      - Randomly sample a proportion (SAMPLE_RATE) of eligible segments.
      - For each candidate, attempt to merge it with its following segment.
      - If the rewriting process meets the BLEURT threshold within MAX_ATTEMPTS, update the candidate segment by
        combining its 'src', concatenating 'tgt', and setting seg_id to "oldID_nextID". Then mark the following segment
        for removal.
      - If not, skip this candidate and try another.
    
    After processing, the merged segments are kept and the original segments that were merged (the second parts) are dropped.
    
    Returns the updated list of segments.
    """
    updated_docs = {}
    merge_prompt_template = (
    "Please merge these two segments into one sentence while preserving their original meaning, word choice, and order. "
    "Instead of simply concatenating them, "
    "use appropriate transitional expressions so that the segments are naturally connected without merely inserting a period or extra whitespace. "
    "Ensure the final result flows coherently and no important information is omitted. "
    "Return only the merged text on a single line, with no additional commentary or extraneous text.\n\n"
    "First segment:\n{first_sent}\n\n"
    "Second segment:\n{second_sent}\n\n"
    )
    
    for doc_id, segments in docs.items():
        doc_segments = deepcopy(segments)
        num_segments = len(doc_segments)
        if num_segments < 2:
            updated_docs[doc_id] = doc_segments
            continue

        # Eligible segments: not the first and not the last
        eligible_indices = list(range(1, num_segments - 1))
        random.shuffle(eligible_indices)
        desired_ops = max(1, math.ceil(len(eligible_indices) * sample_rate))
        processed_ops = 0
        used_indices = set()
        
        for idx in eligible_indices:
            if idx in used_indices or (idx + 1) in used_indices:
                continue

            # Ensure the next segment exists
            if idx + 1 >= len(doc_segments):
                continue

            current_seg = doc_segments[idx]
            next_seg = doc_segments[idx + 1]
            combined_original = current_seg["src"] + " " + next_seg["src"]
            prompt = merge_prompt_template.format(
                first_sent=current_seg["src"],
                second_sent=next_seg["src"]
            )
            attempts = []
            rewrite_count = 0
            success = False
            while rewrite_count < MAX_ATTEMPTS:
                candidate = call_gpt4_api(prompt)
                rewrite_count += 1
                score_val = bleurt_score(combined_original, candidate)
                print("\n=====================================")
                print(current_seg["src"])
                print()
                print(next_seg["src"])
                print()
                print(candidate)
                print()
                print(score_val)
                print("\n=====================================")
                if LOG_REWRITING:
                    attempts.append({
                        "attempt": rewrite_count,
                        "candidate": candidate,
                        "bleurt_score": score_val
                    })
                if score_val >= bleurt_threshold:
                    merged_src = candidate
                    success = True
                    break
            if not success:
                continue

            # Update the current segment with the merged result
            current_seg["src"] = merged_src
            current_seg["tgt"] = current_seg["tgt"] + " " + next_seg["tgt"]
            current_seg["seg_id"] = f"{current_seg['seg_id']}_{next_seg['seg_id']}"
            used_indices.add(idx)
            used_indices.add(idx + 1)
            # Mark the next segment for removal by setting it to None
            doc_segments[idx + 1] = None

            if LOG_REWRITING:
                log_entry = {
                    "operation": "merge",
                    "doc_id": doc_id,
                    "seg_ids": [current_seg["seg_id"], next_seg["seg_id"]],
                    "prompt": prompt,
                    "original_text": combined_original,
                    "final_text": merged_src,
                    "attempts": attempts
                }
                rewriting_logs.append(log_entry)
            processed_ops += 1
            if processed_ops >= desired_ops:
                break

        # Remove segments that were merged (i.e. marked as None)
        doc_segments = [seg for seg in doc_segments if seg is not None]
        updated_docs[doc_id] = doc_segments

    merged_data = []
    for segs in updated_docs.values():
        merged_data.extend(segs)
    return merged_data

def process_drop_src(docs, sample_rate):
    """
    Process drop_src operation:
    - Randomly select segments but avoid sampling from docs with only one segment
    - Set src to empty string in selected segments
    - Also set tgt to empty string in ref_A (but not in other files)
    """
    all_segments = []
    eligible_indices = []
    
    # First collect all segments and identify eligible ones for dropping
    index = 0
    for doc_id, segments in docs.items():
        for seg in segments:
            all_segments.append(seg)
            # Only consider segments from docs with more than one segment
            if len(segments) > 1:
                eligible_indices.append(index)
            index += 1
    
    # Sample segments to drop
    if not eligible_indices:
        return all_segments  # No eligible segments to drop
        
    num_to_drop = max(1, math.ceil(len(eligible_indices) * sample_rate))
    drop_indices = random.sample(eligible_indices, min(num_to_drop, len(eligible_indices)))
    
    # Drop src and tgt in selected segments
    for idx in drop_indices:
        all_segments[idx]["src"] = ""
        all_segments[idx]["tgt"] = ""  # Also drop tgt in ref_A
        
    return all_segments

def process_drop_tgt(docs, sample_rate):
    """
    Process drop_tgt operation:
    - Randomly select segments but avoid sampling from docs with only one segment
    - For ref_A: Keep the original (no changes)
    - For other files: Create a copy with tgt set to empty string in selected segments
    """
    all_segments = []
    eligible_indices = []
    
    # First collect all segments and identify eligible ones for dropping
    index = 0
    for doc_id, segments in docs.items():
        for seg in segments:
            all_segments.append(seg)
            # Only consider segments from docs with more than one segment
            if len(segments) > 1:
                eligible_indices.append(index)
            index += 1
    
    # Make a copy of the data for comparison with other files
    # This will be the output to use for updating other files
    dropped_segments = deepcopy(all_segments)
    
    # Sample segments to drop
    if not eligible_indices:
        return all_segments, dropped_segments  # No eligible segments to drop
        
    num_to_drop = max(1, math.ceil(len(eligible_indices) * sample_rate))
    drop_indices = random.sample(eligible_indices, min(num_to_drop, len(eligible_indices)))
    
    # Drop tgt in the copy but leave ref_A unchanged
    for idx in drop_indices:
        dropped_segments[idx]["tgt"] = ""  # Only drop tgt in the copy for other files
        
    return all_segments, dropped_segments


def update_other_file(original_data, new_ref_data, op, dropped_data=None):
    """
    Update the other JSONL files in the same folder based on the newly generated ref_A (new_ref_data).

    For merge:
      - If a merged segment in new_ref_data has a composite seg_id (e.g., "14_15"), then in the original_data,
        locate segments with seg_id "14" and "15" (for the same doc_id). Update their src with the merged src
        from new_ref_data, and update their tgt by concatenating the original tgt values (in order).
      - If a segment in new_ref_data has a non-composite seg_id, simply update the matching segment’s src.
    
    For drop_src:
      - For each segment in new_ref_data, if its src is an empty string, locate the corresponding segment
        in original_data and set its src to an empty string (but leave tgt unchanged).
    
    For drop_tgt:
      - Use the dropped_data parameter (which contains segments with tgt dropped) to update the other files.
    """
    if op == "merge":
        for new_seg in new_ref_data:
            doc_id = new_seg["doc_id"]
            new_seg_id = str(new_seg["seg_id"])  # Convert to string in case it's an int.
            if "_" in new_seg_id:
                # Merged segment: e.g., "14_15"
                parts = new_seg_id.split("_")
                # Locate original segments with seg_id in parts (using string conversion)
                original_segments = [
                    seg for seg in original_data
                    if seg["doc_id"] == doc_id and str(seg["seg_id"]) in parts
                ]
                if original_segments:
                    # Concatenate tgt values in order of parts
                    merged_tgt_list = []
                    for part in parts:
                        for seg in original_segments:
                            if str(seg["seg_id"]) == part:
                                merged_tgt_list.append(seg.get("tgt", ""))
                                break
                    merged_tgt = " ".join(merged_tgt_list).strip()
                    # Update each original segment with merged src and tgt
                    for seg in original_segments:
                        seg["src"] = new_seg["src"]
                        seg["tgt"] = merged_tgt
            else:
                # Non-merged segment: update matching segment's src
                for seg in original_data:
                    if seg["doc_id"] == doc_id and str(seg["seg_id"]) == new_seg_id:
                        seg["src"] = new_seg["src"]
        return original_data

    elif op == "drop_src":
        # For drop_src: update only the src field if src is empty in new_ref_data
        for new_seg in new_ref_data:
            doc_id = new_seg["doc_id"]
            new_seg_id = new_seg["seg_id"]
            if new_seg.get("src", None) == "":
                for seg in original_data:
                    if seg["doc_id"] == doc_id and seg["seg_id"] == new_seg_id:
                        seg["src"] = ""  # Only update src, not tgt
        return original_data

    elif op == "drop_tgt":
        # For drop_tgt: Use the dropped_data to update other files
        if dropped_data:
            for i, dropped_seg in enumerate(dropped_data):
                doc_id = dropped_seg["doc_id"]
                seg_id = dropped_seg["seg_id"]
                
                # Only apply changes if tgt is empty in dropped_data
                if dropped_seg.get("tgt", None) == "":
                    for seg in original_data:
                        if seg["doc_id"] == doc_id and seg["seg_id"] == seg_id:
                            seg["tgt"] = ""
        return original_data

    else:
        return original_data



def main():
    # List of folders to process
    input_folders = [
        "data/wmt24/json_output_en_de",
        "data/wmt24/json_output_en_es",
        "data/wmt24/json_output_ja_zh"
    ]
    
    # Define operations and for drop operations, indicate drop_side
    operations = [
        "merge",
        "drop_src",
        "drop_tgt"
    ]
    
    for op in operations:
        for folder in input_folders:
            folder_name = os.path.basename(folder)
            output_subdir = os.path.join(folder, op)
            os.makedirs(output_subdir, exist_ok=True)
            
            ref_A_path = os.path.join(folder, "ref_A.jsonl")
            if not os.path.exists(ref_A_path):
                print(f"Warning: {ref_A_path} does not exist. Skipping folder {folder_name}.")
                continue
            
            ref_A_data = load_jsonl(ref_A_path)
            docs = group_by_doc(ref_A_data)
            
            if op == "merge":
                BLEURT_THRESHOLD = 0.85
                new_ref_data = process_merge(docs, SAMPLE_RATE, BLEURT_THRESHOLD)
                dropped_data = None
            elif op == "drop_src":
                new_ref_data = process_drop_src(docs, SAMPLE_RATE)
                dropped_data = None
            elif op == "drop_tgt":
                new_ref_data, dropped_data = process_drop_tgt(docs, SAMPLE_RATE)
            else:
                continue
            
            new_ref_A_path = os.path.join(output_subdir, "ref_A.jsonl")
            write_jsonl(new_ref_A_path, new_ref_data)
            print(f"[{op}] Processed ref_A.jsonl for folder {folder_name} -> {new_ref_A_path}")
            
            for filename in os.listdir(folder):
                if filename.endswith(".jsonl") and filename not in ["ref_A.jsonl", "ref_B.jsonl"]:
                    orig_file_path = os.path.join(folder, filename)
                    original_data = load_jsonl(orig_file_path)
                    updated_data = update_other_file(original_data, new_ref_data, op, dropped_data)
                    new_file_path = os.path.join(output_subdir, filename)
                    write_jsonl(new_file_path, updated_data)
                    print(f"[{op}] Updated file written to {new_file_path}")

if __name__ == "__main__":
    main()
