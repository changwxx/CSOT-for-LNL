# [NeurIPS 2023] CSOT: Curriculum and Structure-Aware Optimal Transport for Learning with Noisy Labels
Code release for CSOT: Curriculum and Structure-Aware Optimal Transport for Learning with Noisy Labels (NeurIPS 2023).

[[paper]](https://openreview.net/forum?id=y50AnAbKp1) [[project page]](https://changwxx.github.io/CSOT-webpage/)

## Requirements
* Python 3.7+
* PyTorch 1.8.0
* GPU Memory 24+ GB
> We have conducted our experiments on a single GPU of NVIDIA A100 with 80 GB memory.
> We follow [DivideMix](https://github.com/LiJunnan1992/DivideMix) and [NCE](https://github.com/lijichang/LNL-NCE) to construct our codebase.

## Getting started
* Modify data_path in main.py
* Train with command line 
    ```
    CUDA_VISIBLE_DEVICES=0 python main_cifar.py
    ```

## Cite our work
If you find this repository useful in your research, please consider citing:
```
@inproceedings{
    chang2023csot,
    title={{CSOT}: Curriculum and Structure-Aware Optimal Transport for Learning with Noisy Labels},
    author={Wanxing Chang and Ye Shi and Jingya Wang},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=y50AnAbKp1}
}
```