# Getting Started with PrismRCL

PrismRCL is a Windows-based AI classification algorithm that supports image, text, and tabular data classification. This guide covers all features and provides examples for each data type.

## Table of Contents
- [Installation](#installation)
- [Dataset Requirements](#dataset-requirements)
- [Evaluation Methods](#evaluation-methods)
- [Working with Images](#working-with-images)
- [Working with Text (LLM)](#working-with-text-llm)
- [Working with Tabular Data](#working-with-tabular-data)
- [Auto-Optimization](#auto-optimization)
- [Transfer Learning](#transfer-learning)
- [Web-Based Inference](#web-based-inference)
- [Model Management](#model-management)
- [Enterprise Features](#enterprise-features)

## Installation

1. Download the PrismRCL package
2. Unzip the files into a directory on your hard drive
3. No additional installation is required

Note: PrismRCL is a command-line tool designed for developers to build applications and scripts. It doesn't include an interactive user interface.

## Dataset Requirements

PrismRCL supports three types of data:
- Images (PNG format)
- Text data (individual text files)
- Tabular data (space-separated values in single-line text files)

### Required Directory Structure

Your dataset must be organized in the following structure:
```
parent_folder/
    class1_folder/
        sample1.png (or .txt)
        sample2.png (or .txt)
        ...
    class2_folder/
        sample1.png (or .txt)
        sample2.png (or .txt)
        ...
```

Important: File names must be unique across all class folders.

## Evaluation Methods

PrismRCL offers three evaluation methods:
- `naivebayes`: Required for LLM/text classification
- `fractal`: Optimized for image classification
- `chisquared`: Suitable for both image and tabular data

## Working with Images

### Example Datasets
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 60,000 32x32 color images in 10 classes
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): 70,000 28x28 grayscale images of clothing items
- [Flowers Recognition](https://www.kaggle.com/alxmamaev/flowers-recognition): 4,242 images of flowers in 5 categories

### Training an Image Classifier

```bash
PrismRCL.exe fractal rclicks=15 data="C:\PrismRCL\data\flower-dataset\train-data" testsize=0.1 savemodel="C:\PrismRCL\models\flower-classifier.classify" log="C:\PrismRCL\logfiles\job_folder" stopwhendone
```

Parameters explained:
- `fractal`: Evaluation method optimized for images
- `rclicks`: Training parameter (recommended range: 15-20 for images)
- `testsize`: Fraction of data used for testing
- `savemodel`: Output model path
- `log`: Log directory

## Working with Text (LLM)

### Example Datasets
- [IMDB Reviews](https://ai.stanford.edu/~amaas/data/sentiment/): 50,000 movie reviews for sentiment analysis
- [AG News](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html): News articles categorized by topic
- [SMS Spam Collection](https://www.kaggle.com/uciml/sms-spam-collection-dataset): SMS messages labeled as spam/ham

### Training a Text Classifier

```python
import os

# Base directory - change this to your desired location
base_dir = r"C:\Users\sam\Desktop\lumina\prism-rcl-ent-v2.6.0"

# Create directory structure
train_dir = os.path.join(base_dir, "data", "llm-dataset", "train-data")
test_dir = os.path.join(base_dir, "data", "llm-dataset", "test-data")
model_dir = os.path.join(base_dir, "models")
log_dir = os.path.join(base_dir, "logfiles", "job_folder")

# Create all necessary directories
for directory in [
    os.path.join(train_dir, "positive"),
    os.path.join(train_dir, "negative"),
    os.path.join(test_dir, "positive"),
    os.path.join(test_dir, "negative"),
    model_dir,
    log_dir
]:
    os.makedirs(directory, exist_ok=True)

# Training data
train_positive = {
    "review1.txt": "This movie was absolutely fantastic! The acting was superb.",
    "review2.txt": "I loved every minute of this film. Great storyline and amazing effects.",
    "review3.txt": "One of the best movies I've seen this year. Highly recommended!",
    "review4.txt": "Outstanding performance by the entire cast. A must-watch!",
    "review5.txt": "Brilliant direction and engaging plot from start to finish."
}

train_negative = {
    "review1.txt": "I couldn't wait for this movie to end. Very disappointing.",
    "review2.txt": "Poor writing and terrible pacing throughout the film.",
    "review3.txt": "Save your money and skip this one. Not worth watching.",
    "review4.txt": "The worst movie I've seen in years. Complete waste of time.",
    "review5.txt": "Terrible acting and predictable plot. Would not recommend."
}

# Test data
test_positive = {
    "test1.txt": "An incredible cinematic experience. I was blown away!",
    "test2.txt": "Such a wonderful film with amazing performances."
}

test_negative = {
    "test1.txt": "Boring and poorly executed. Don't bother watching.",
    "test2.txt": "Nothing original about this movie. Very disappointing."
}

# Function to write files
def write_files(directory, files_dict):
    for filename, content in files_dict.items():
        with open(os.path.join(directory, filename), 'w') as f:
            f.write(content)

# Write all files
write_files(os.path.join(train_dir, "positive"), train_positive)
write_files(os.path.join(train_dir, "negative"), train_negative)
write_files(os.path.join(test_dir, "positive"), test_positive)
write_files(os.path.join(test_dir, "negative"), test_negative)

print("Dataset creation complete!")
print("\nCommand to run training:")
print(f'"{os.path.join(base_dir, "PrismRCL.exe")}" llm naivebayes directional rclicks=67 readtextbyline ' + 
      f'data="{train_dir}" ' +
      f'testdata="{test_dir}" ' +
      f'savemodel="{os.path.join(model_dir, "sentiment-model.classify")}" ' +
      f'log="{log_dir}" stopwhendone')
```

Parameters explained:
- `llm`: Indicates language model training
- `naivebayes`: Required evaluation method for text
- `directional`: Maintains word order
- `rclicks=67`: Recommended for text processing
- `readtextbyline`: Process text line by line

## Working with Tabular Data

### Example Datasets
- [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris): 150 samples of iris flowers with 4 features
- [Wisconsin Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29): 569 samples with 30 features
- [Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality): 4,898 wine samples with 11 features

### Training a Tabular Classifier

```bash
PrismRCL.exe chisquared data="C:\PrismRCL\data\tabular-dataset\train-data" testsize=0.2 savemodel="C:\PrismRCL\models\tabular-classifier.classify" log="C:\PrismRCL\logfiles\job_folder" stopwhendone
```

Note: Starting from version 2.4.0, tabular data values no longer need to be normalized.

## Auto-Optimization

PrismRCL can automatically optimize training parameters:

```bash
PrismRCL.exe auto-optimize data="C:\PrismRCL\data\train-data" log="C:\PrismRCL\log_files\"
```

Optimization goals are selected based on your dataset:
- Two classes: Overall Accuracy (acc)
- Balanced dataset: Macro Average F1 Score (af1)
- Imbalanced dataset: Weighted Average F1 Score (wf1)
- Custom: Matthews Correlation Coefficient (mcc)

```bash
# Force specific optimization goal
PrismRCL.exe auto-optimize=mcc data="C:\PrismRCL\data\train-data" log="C:\PrismRCL\log_files\"
```

## Transfer Learning

Transfer learning allows you to build upon pre-trained models:

```bash
PrismRCL.exe naivebayes rclicks=15 transferlearn="C:\PrismRCL\models\base_model.classify" data="C:\PrismRCL\data\dataset\train-data" testsize=0.1 savemodel="C:\PrismRCL\models\enhanced_model.classify" log="C:\PrismRCL\logfiles\job_folder" stopwhendone
```

## Model Management

### Combining Models

```bash
PrismRCL.exe loadmodel="C:\PrismRCL\models\model1.classify" addmodel="C:\PrismRCL\models\model2.classify;C:\PrismRCL\models\model3.classify" savemodel="C:\PrismRCL\models\combined_model.classify" stopwhendone
```

Note: Combined models must be trained with the same parameters and on similar data.

## Web-Based Inference

Start a web inference server:

```bash
PrismRCLM.exe loadmodel="C:\models\model111.classify" port=8080
```

Access via HTTP:
```
http://localhost:8080/evaluation_method&imaginary_flag&testdata=C:/RCLC/data/test_data/&log=C:/RCLC/logfiles/&inftotext=C:/RCLC/output/prediction_output.txt
```

Server status responses:
- `ready`: Server idle
- `done`: Request processed
- `busy`: Processing another request
- `error`: Request error

## Enterprise Features

- Docker container support
- Multiple model loading on different ports
- Network drive support for web inference
- Advanced logging and monitoring
- Automated deployment tools

## Version Compatibility

- Version 2.6.0 models are not backward compatible
- Models from earlier versions are not compatible with 2.6.0
- Docker support requires Enterprise license

## Tips

1. Use descriptive model names: `dataset-type_v260_date.classify`
2. Match evaluation methods to data types:
   - Images: `fractal` or `chisquared`
   - Text: `naivebayes` (required)
   - Tabular: `chisquared` (recommended)
3. Training parameters are saved in model files
4. Use auto-optimize for best results with new datasets

## License

PrismRCL includes a 30-day trial period. Purchase a subscription and activate using the provided license key for continued use.

## Support

For additional support and documentation, contact our support team or visit our documentation portal.
