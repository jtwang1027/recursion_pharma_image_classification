Image classification using on [Rxrx1 dataset](https://www.rxrx.ai/rxrx1)


```
python train.py --config_path example_config.yaml
```

Use the config to specify model architecture (densenet or ViT transformer) and training parameters (num epochs, data augmentation methods, learning rate and schedulers.) The loss function is based on a joint Arcface and cross entropy loss weighted by a coefficient (`loss_ce_weight`).


Densenet is implementation is based on this [winning version](https://github.com/maciej-sypetkowski/kaggle-rcic-1st) from [kaggle](https://www.kaggle.com/competitions/recursion-cellular-image-classification/overview).

This repo is structured in a way to test out various hyperparameters, data augmentation methods, and compare densenet vs ViT on this dataset. To load the dataset, the [metadata file](https://storage.googleapis.com/rxrx/rxrx1/rxrx1-metadata.zip) and [images_dir](https://storage.googleapis.com/rxrx/rxrx1/rxrx1-images.zip) need to be downloaded. Note that the images_dir is 45GB, and could be loaded in a [kaggle environment](https://www.kaggle.com/competitions/recursion-cellular-image-classification/overview) (Code> +New Notebook).

