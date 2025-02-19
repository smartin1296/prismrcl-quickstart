# Getting Started with PrismRCL

PrismRCL is a Windows-based AI classification algorithm that supports image, text, and tabular data classification. This guide will help you get started with basic setup and usage.

## Installation

1. Download the PrismRCL package at https://lumina247.com/prismrcl-sign-up/
2. Unzip the files into a directory on your hard drive
3. No additional installation is required

Note: PrismRCL is a command-line tool designed for developers to build applications and scripts. It doesn't include an interactive user interface.

## Dataset Requirements

PrismRCL supports three types of data:
- Images (PNG format)
- Text data
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

## Quick Start Example: Text Classification

Here's a complete example of setting up and running a simple sentiment analysis model using PrismRCL.

### 1. Create Dataset Structure

```python
import os

# Base directory - change this to your PrismRCL installation path
base_dir = r"C:\PrismRCL"

# Create directory structure
train_dir = os.path.join(base_dir, "data", "llm-dataset", "train-data")
test_dir = os.path.join(base_dir, "data", "llm-dataset", "test-data")
model_dir = os.path.join(base_dir, "models")
log_dir = os.path.join(base_dir, "logfiles", "job_folder")

# Create directories
for directory in [
    os.path.join(train_dir, "positive"),
    os.path.join(train_dir, "negative"),
    os.path.join(test_dir, "positive"),
    os.path.join(test_dir, "negative"),
    model_dir,
    log_dir
]:
    os.makedirs(directory, exist_ok=True)

# Sample training data
train_positive = {
    "review1.txt": "This movie was absolutely fantastic! The acting was superb.",
    "review2.txt": "I loved every minute of this film. Great storyline."
}

train_negative = {
    "review1.txt": "I couldn't wait for this movie to end. Very disappointing.",
    "review2.txt": "Poor writing and terrible pacing throughout the film."
}

# Sample test data
test_positive = {
    "test1.txt": "An incredible cinematic experience. I was blown away!"
}

test_negative = {
    "test1.txt": "Boring and poorly executed. Don't bother watching."
}

# Write files function
def write_files(directory, files_dict):
    for filename, content in files_dict.items():
        with open(os.path.join(directory, filename), 'w') as f:
            f.write(content)

# Create all files
write_files(os.path.join(train_dir, "positive"), train_positive)
write_files(os.path.join(train_dir, "negative"), train_negative)
write_files(os.path.join(test_dir, "positive"), test_positive)
write_files(os.path.join(test_dir, "negative"), test_negative)
```

### 2. Train the Model

Run the following command:

```bash
PrismRCL.exe llm naivebayes directional rclicks=67 readtextbyline data="C:\PrismRCL\data\llm-dataset\train-data" testdata="C:\PrismRCL\data\llm-dataset\test-data" savemodel="C:\PrismRCL\models\sentiment-model.classify" log="C:\PrismRCL\logfiles\job_folder" stopwhendone
```

Command parameters explained:
- `llm`: Indicates language model training
- `naivebayes`: Evaluation method (required for LLM)
- `directional`: Maintains word order in text
- `rclicks=67`: Training parameter
- `readtextbyline`: Processes text input line by line
- `data`: Path to training data
- `testdata`: Path to test data
- `savemodel`: Where to save the trained model
- `log`: Directory for log files
- `stopwhendone`: Automatically shutdown after completion

## Auto-Optimization

PrismRCL includes an auto-optimize feature to find optimal training parameters:

```bash
PrismRCL.exe auto-optimize data="C:\PrismRCL\data\train-data" log="C:\PrismRCL\log_files\"
```

Auto-optimize goals are automatically selected based on your dataset:
- Two classes: Overall Accuracy (acc)
- Balanced dataset: Macro Average F1 Score (af1)
- Imbalanced dataset: Weighted Average F1 Score (wf1)

You can also specify a goal:
```bash
PrismRCL.exe auto-optimize=mcc data="C:\PrismRCL\data\train-data" log="C:\PrismRCL\log_files\"
```

## Web-Based Inference

PrismRCL supports web-based inference through PrismRCLM.exe:

```bash
PrismRCLM.exe loadmodel="C:\models\model111.classify" port=8080
```

Access the model via HTTP requests:
```
http://localhost:8080/evaluation_method&imaginary_flag&testdata=C:/RCLC/data/test_data/&log=C:/RCLC/logfiles/&inftotext=C:/RCLC/output/prediction_output.txt
```

## Version Compatibility

- Version 2.6.0 models are not backward compatible
- Models from earlier versions are not compatible with 2.6.0

## Enterprise Features

- Docker container support (Enterprise clients only)
- Multiple model loading on different ports
- Network drive support for web inference

## Tips

1. Use descriptive model names, e.g., `sentiment-model_v260_01102025.classify`
2. For LLM training, always use:
   - `naivebayes` evaluation method
   - `directional` parameter
3. Training parameters are saved in the model file and loaded automatically during inference

## License

PrismRCL comes with a 30-day trial period. For continued use, purchase a subscription and activate using the provided license key.
