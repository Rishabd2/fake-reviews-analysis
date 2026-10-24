# Sources and licenses

## Code

Bag-of-words, linguistic features, n-grams, and BERT training code were adapted from [PauDK/Deceptive-Review-Detection](https://github.com/PauDK/Deceptive-Review-Detection). That repository had no license file when this repo was assembled. The generation stack (`generator.py`, `review_eval.py`, `main.py`, `archive/llm2.py`) and the plotting scripts are original to this project.

This repository is MIT-licensed for the original files. If you reuse the PauDK modules, check that project for any later license change.

## Data

Deceptive Opinion Spam Corpus v1.4, Myle Ott et al., [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). Not redistributed here.

## Papers

- Ott, M., Choi, Y., Cardie, C., and Hancock, J. T. (2011). Finding deceptive opinion spam by any stretch of the imagination. ACL-HLT. https://arxiv.org/abs/1107.4557
- Ott, M., Cardie, C., and Hancock, J. T. (2013). Negative deceptive opinion spam. NAACL-HLT. https://aclanthology.org/N13-1053.pdf
- Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT. https://arxiv.org/abs/1810.04805
- Paul, H., and Nikolaev, A. (2021). Fake review detection on online E-commerce platforms: a systematic literature review. Data Mining and Knowledge Discovery 35(5).

Local copies of the Ott PDFs sat in `Downloads/adv_fake reviews/` during assembly. They are not copied into this repo.
