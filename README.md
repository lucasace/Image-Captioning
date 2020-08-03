# Image Captioning

## Dataset Preparation
* Download the Flickr8k Image and Text dataset from ![here](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip) and ![here](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip) respectively
* Unzip both the dataset and text files
* To extract image_features and to preprocess the captions
  ```bash
  python3 dataset.py
  ```
  You can make changes to the arguments of the class DataManager in dataset.py
