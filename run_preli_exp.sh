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

# en-de
python preli_exp.py --input_dir data/wmt24/json_output_en_de/raw --scenario over
python preli_exp.py --input_dir data/wmt24/json_output_en_de/raw --scenario under

# en-es
python preli_exp.py --input_dir data/wmt24/json_output_en_es/raw --scenario over
python preli_exp.py --input_dir data/wmt24/json_output_en_es/raw --scenario under

# ja-zh
python preli_exp.py --input_dir data/wmt24/json_output_ja_zh/raw --scenario over
python preli_exp.py --input_dir data/wmt24/json_output_ja_zh/raw --scenario under
