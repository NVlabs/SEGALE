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

# v0.1

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr python3.10-dev ffmpeg curl  # use python 3.10
RUN ln -s /usr/bin/python3.10 /usr/bin/python3
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && python3 -m pip --version  # use python 3.10 pip
RUN python3 -m pip install --no-cache-dir pip==24.0  # pip 24.0 is required by LASER

ARG PYTORCH='2.5.0'
ARG TORCH_VISION=''
ARG TORCH_AUDIO=''
ARG CUDA='cu121'

# actual docker file

RUN apt-get install -y wget unzip vim python-is-python3 cmake

RUN pip install spacy==3.8.4 pandas==2.2.3 datasets==3.4.1 accelerate==0.28.0 unbabel-comet==2.2.2 mcerp
RUN git clone https://github.com/facebookresearch/LASER /opt/LASER && cd /opt/LASER && pip install -e .
ENV LASER=/opt/LASER
RUN cd /opt/LASER && bash ./nllb/download_models.sh ace_Latn && bash ./install_external_tools.sh && sed -i 's#model_dir=""#model_dir="/opt/LASER"#' /opt/LASER/tasks/embed/embed.sh

# running this late after laser to ensure torch version won't be overwritten 
RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python -m spacy download en_core_web_sm
RUN python -m spacy download de_core_news_sm
RUN python -m spacy download zh_core_web_sm
RUN python -m spacy download ja_core_news_sm
RUN python -m spacy download es_core_news_sm

# TODO: add your HF_TOKEN
ENV HF_TOKEN=""
RUN pip install -U "huggingface_hub[cli]"
RUN python -c "from huggingface_hub._login import _login ; _login(token=\"$HF_TOKEN\", add_to_git_credential=False)"

# ersatz
RUN pip install sentencepiece==0.1.95 tensorboard==2.19.0 progressbar2
RUN git clone https://github.com/rewicks/ersatz /opt/ersatz && cd /opt/ersatz && sed -i 's#torch==1.7.1#torch#' setup.py && sed -i 's#tensorboard==2.4.1#tensorboard==2.19.0#' setup.py && python setup.py install

RUN pip install -U transformers && pip install protobuf==3.20  # metricx doesn't work with existing transformers package
