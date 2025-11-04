# SEGALE 

SEGALE is a tool that allows for the extension of existing sentence-level machine translation metrics to document-level machine translation.
Functionally, it is similar to [mwerSegmenter](https://github.com/cservan/MWERalign), which has been used as the long-standing standard for [IWSLT evaluations](https://iwslt.org/), but offers the following additional benefits:

- More robust performance when encountering over-/under-translation errors
- Does not depend on a reference translation to operate

If you use this tool for your work, please cite the following paper:

```
@inproceedings{wang-etal-2025-extending,
    title = "Extending Automatic Machine Translation Evaluation to Book-Length Documents",
    author = "Wang, Kuang-Da  and
      Ding, Shuoyang  and
      Yang, Chao-Han Huck  and
      Hsieh, Ping-Chun  and
      Peng, Wen-Chih  and
      Lavrukhin, Vitaly  and
      Ginsburg, Boris",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1645/",
    pages = "32311--32327",
    ISBN = "979-8-89176-332-6"
}
```

## Install

The best way to reproduce our experiment environment to the maximum extent possible is to rebuild the docker container with the Dockerfile and run everything inside Docker, but installing this package in other environments should also work.

Depending on how you install, you may want to make some of the following edits:

+ Add `HF_TOKEN` to the `Dockerfile`
+ Set up [LASER](https://github.com/facebookresearch/LASER) and edit `LASER_DIR` in `segale_align.py`. If you use our docker image, it's already setup for you.

Installation itself is very easy:

```
git clone --recurse-submodules https://github.com/nvlabs/SEGALE
cd SEGALE
pip install -e .
```

This will add two new commands in your workspace: `segale-align` and `segale-eval`.

To run `segale-eval`, you should download the following models:

```
huggingface-cli download google/metricx-24-hybrid-large-v2p6
huggingface-cli download Unbabel/wmt22-comet-da
huggingface-cli download Unbabel/wmt22-cometkiwi-da
```

## Usage

We'll introduce the usage of those commands using the WMT 2024 metrics shared task data as example. You can access our reformatted and augmented WMT24 dataset [here](https://huggingface.co/datasets/rl-bandits-lab/SEGALE-WMT24). To proceed with the subsequent steps, you need to download the dataset inside this repo:

```
mkdir -p data
git clone https://huggingface.co/datasets/rl-bandits-lab/SEGALE-WMT24 data
```

### Step 1: Src-Ref-Tgt Alignment (`segale-align`)

To align a system output file (tgt) with the source (src) and the segmented reference (ref), use the following command. For example, given a system file like `data/wmt24/json_output_ja_zh/raw/GPT-4.jsonl`, and a reference file like `spacy_ref_A.jsonl`, run:

```bash
segale-align --system_file data/wmt24/json_output_ja_zh/raw/GPT-4.jsonl --ref_file data/wmt24/json_output_ja_zh/raw/spacy_ref_A.jsonl --segmenter spacy --task_lang zh --proc_device cuda -v
```

> - `--segmenter`: Choose between `spacy` or `ersatz`.
> - `--task_lang`: Required for spaCy segmentation to specify the target language (e.g., `zh`).
> - `--proc_device`: Specify `cuda` or `cpu`, depending on GPU availability.
> - `-v` / `-vv`: Set verbosity level.  
>    - `-v`: Saves the intermediate results of the adaptive penalty search process.  
>    - `-vv`: Additionally saves individual alignment results for each document.  
>    - If not set, only the final system-level alignment result will be saved.

The aligned output will be stored in a folder corresponding to the system file, e.g., `data/wmt24/json_output_ja_zh/raw/GPT-4/`, with the key file being:

```
aligned_spacy_GPT-4.jsonl
```

This file is used for subsequent evaluation.

### Step 2: Evaluation (`segale-eval`)

This script runs 

Once alignment is complete, you can evaluate the aligned file using:

```bash
segale-eval --input_file data/wmt24/json_output_ja_zh/raw/GPT-4/aligned_spacy_GPT-4.jsonl
```

This will generate:
- `eval_aligned_spacy_GPT-4.jsonl`: Evaluation results.
- `result_aligned_spacy_GPT-4.jsonl`: Document-level aggregated results.

## Reproducing Experimental Results in the Paper

Before you start, follow the instructions under "Usage" section to checkout the necessary datasets.

### Automate Experiments

To generate all alignment and evaluation commands for multiple system files, use:

```bash
python generate_eval_script.py
```

This will generate a script named `run_eval.sh`, which can be executed to perform batch alignment and evaluation across all system outputs.

### Generate Sanity Check Dataset

This script enables the simulation of over-translation, under-translation, and sentence boundary alterations in machine translation outputs. It combines merging or dropping operations using GPT-4 API and BLEURT checks, and supports batch processing over multiple folders.

```bash
python gen_sanity_check_dataset.py
```

### Source-Reference Alignment

To perform alignment between source and reference (src-ref), you can start with the default boundary file `ref_A.jsonl`.

Since estimating suitable alignment parameters is required for adaptive penalty search, please run the following script first:

```bash
./gen_ref_paras.sh
```

The estimated parameters will be saved inside the `ref_A` folder.

You can also use either **spaCy** or **ersatz** to perform automatic sentence segmentation and alignment (e.g., when the reference file does not contain boundary information):

```bash
./gen_aligned_ref.sh
```

After execution, you will obtain the aligned files `spacy_ref_A.jsonl` or `ersatz_ref_A.jsonl`. The corresponding alignment parameters will also be saved in the `spacy_ref_A` or `ersatz_ref_A` folders respectively.
