# SEGALE

SEGALE is a tool that allows for the extension of existing sentence-level machine translation metrics to document-level machine translation. Functionally, it is similar to [mwerSegmenter](https://github.com/cservan/MWERalign), which has been used as the long-standing standard for [IWSLT evaluations](https://iwslt.org/), but offers the following additional benefits:

- More robust performance when encountering over-/under-translation errors
- Does not depend on a reference translation to operate

We will release the code when it's ready. Until then, please refer to [our paper](https://arxiv.org/abs/2509.17249).

```
@misc{wang2025extendingautomaticmachinetranslation,
      title={Extending Automatic Machine Translation Evaluation to Book-Length Documents}, 
      author={Kuang-Da Wang and Shuoyang Ding and Chao-Han Huck Yang and Ping-Chun Hsieh and Wen-Chih Peng and Vitaly Lavrukhin and Boris Ginsburg},
      year={2025},
      eprint={2509.17249},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.17249}, 
}
```
