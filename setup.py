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

from setuptools import setup, find_packages

setup(
    name='segale',
    version='0.1.0',
    py_modules=['segale_align', 'segale_eval'],
    packages=find_packages(),
    install_requires=[
        'spacy==3.8.4',
        'torch==2.5.0',
        'numpy==1.23.5',
        'pandas==2.2.3',
        'tqdm==4.67.1',
        'transformers==4.51.3',
        'unbabel-comet==2.2.7',
        'hydra-core==1.3.2',
        'Cython==3.0.11',
    ],
    entry_points={
        'console_scripts': [
            'segale-align = segale_align:main',
            'segale-eval = segale_eval:main',
            'vecalign = vecalign.vecalign:_main',
        ],
    },
)
