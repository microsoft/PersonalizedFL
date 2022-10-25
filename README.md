# PersonalizedFL: Personalized Federated Learning Codebase

An easy-to-learn, easy-to-extend, and for-fair-comparison codebase based on PyTorch for federated learning (FL). 
Please note that this repository is designed mainly for research, and we discard lots of unnecessary extensions for a quick start.
Example usage: when you want to recognize activities of different persons without accessing their privacy data; when you want to build a model based on multiple patients' data but not access their own data.

## Implemented Algorithms

As initial version, we support the following algoirthms. We are working on more algorithms. 

1. Baseline, train in the client without communication.
2. FedAvg [1].
3. FedProx [2].
4. FedBN [3].
5. FedAP [4].
6. MetaFed [5].

## Installation

```
git clone https://github.com/microsoft/PersonalizedFL.git
cd PersonalizedFL
```
We recommend to use `Python 3.7.1` and `torch 1.7.1` which are in our development environment. 
For more environmental details and a full re-production of our results, please refer to `luwang0517/torch10:latest` (docker) or `jindongwang/docker` (docker).

## Dataset

Our code supports the following dataset:

* [VLCS](https://wjdcloud.blob.core.windows.net/dataset/VLCS/VLCS.zip)
* [PACS](https://wjdcloud.blob.core.windows.net/dataset/PACS.zip)
* [Office-Home](https://wjdcloud.blob.core.windows.net/dataset/OfficeHome.zip)
* [PAMAP2](https://wjdcloud.blob.core.windows.net/dataset/cycfed/pamap.tar.gz)
* [COVID](https://wjdcloud.blob.core.windows.net/dataset/cycfed/covid19.tar.gz)
* [OrganS-MNIST](https://wjdcloud.blob.core.windows.net/dataset/cycfed/medmnist.tar.gz)
* [OrganA-MNIST](https://wjdcloud.blob.core.windows.net/dataset/cycfed/medmnistA.tar.gz)
* [OrganC-MNIST](https://wjdcloud.blob.core.windows.net/dataset/cycfed/medmnistC.tar.gz)

If you want to use your own dataset, please modifty `datautil/prepare_data.py` to contain the dataset.

## Usage

1. Modify the file in the scripts
2. `bash run.sh`

## Benchmark

We offer a benchmark for OrganS-MNIST. Please note that the results are based on the data splits in `split/medmnist0.1`. Different data splits may lead different results. For complete parameters, please refer to `run.sh`.

| Non-iid alpha | Base | FedAvg | FedProx | FedBN | FedAP | MetaFed |
|----------|----------|----------|----------|----------|----------|----------|
| 0.1 | 73.99 | 75.62 | 75.97 | 79.96 | 81.33 | 83.87 |
| 0.01 | 75.83 | 74.81 | 75.09 | 81.85 | 82.87 | 84.98 |

## Customization

It is easy to design your own method following the steps:

1. Add your method to `alg/`, and add the reference to it in `alg/algs.py`.

2. Midify `scripts/run.sh` and execuate it.


## Contribution

The toolkit is under active development and contributions are welcome! Feel free to submit issues and PRs to ask questions or contribute your code. If you would like to implement new features, please submit a issue to discuss with us first.

## Reference

[1] McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017.

[2] Li, Tian, et al. "Federated optimization in heterogeneous networks." Proceedings of Machine Learning and Systems 2 (2020): 429-450.

[3] Li, Xiaoxiao, et al. "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization." International Conference on Learning Representations. 2021.

[4] Lu, Wang, et al. "Personalized Federated Learning with Adaptive Batchnorm for Healthcare." IEEE Transactions on Big Data (2022).

[5] Yiqiang, Chen, et al. "MetaFed: Federated Learning among Federations with Cyclic Knowledge Distillation for Personalized Healthcare." FL-IJCAI Workshop 2022.

## Citation

If you think this toolkit or the results are helpful to you and your research, please cite us!

```
@Misc{PersonalizedFL,
howpublished = {\url{https://github.com/microsoft/PersonalizedFL}},   
title = {PersonalizedFL: Personalized Federated Learning Toolkit},  
author = {Lu, Wang and Wang, Jindong}
}  
```

## Contact

- Wang lu: luwang@ict.ac.cn
- [Jindong Wang](http://www.jd92.wang/): jindongwang@outlook.com

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
