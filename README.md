# Getting Started with PrismRCL

## Quick Start: Text Classification in 5 Minutes
Get your first model running quickly with this basic text classification example.

### Prerequisites
- Windows machine with AMD or Intel CPU
- Python
- PrismRCL package ([download here](https://lumina247.com/prismrcl-sign-up/))

### Basic Setup
1. Download and unzip PrismRCL to your hard drive
2. Run activate.exe to start your 30-day trial

### Your First Model
Copy this Python script to create a simple sentiment analysis dataset:

```python
import os

# Change this to your PrismRCL location
base_dir = r"C:\PrismRCL"

# Create required directories
train_dir = os.path.join(base_dir, "data", "llm-dataset", "train-data")
test_dir = os.path.join(base_dir, "data", "llm-dataset", "test-data")
model_dir = os.path.join(base_dir, "models")
log_dir = os.path.join(base_dir, "logfiles", "job_folder")

for directory in [
    os.path.join(train_dir, "positive"),
    os.path.join(train_dir, "negative"),
    os.path.join(test_dir, "positive"),
    os.path.join(test_dir, "negative"),
    model_dir,
    log_dir
]:
    os.makedirs(directory, exist_ok=True)

# Sample data
train_positive = {
    "review1.txt": "This movie was fantastic!",
    "review2.txt": "Great film, loved it."
}

train_negative = {
    "review1.txt": "Terrible movie, very disappointing.",
    "review2.txt": "Don't waste your time."
}

test_positive = {"test1.txt": "Amazing film!"}
test_negative = {"test1.txt": "Boring movie."}

# Create files
def write_files(directory, files_dict):
    for filename, content in files_dict.items():
        with open(os.path.join(directory, filename), 'w') as f:
            f.write(content)

write_files(os.path.join(train_dir, "positive"), train_positive)
write_files(os.path.join(train_dir, "negative"), train_negative)
write_files(os.path.join(test_dir, "positive"), test_positive)
write_files(os.path.join(test_dir, "negative"), test_negative)
```

Then run this command to train your model:
```bash
PrismRCL.exe llm naivebayes directional rclicks=67 readtextbyline data="C:\PrismRCL\data\llm-dataset\train-data" testdata="C:\PrismRCL\data\llm-dataset\test-data" savemodel="C:\PrismRCL\models\sentiment-model.classify" log="C:\PrismRCL\logfiles\job_folder" stopwhendone
```

That's it! You've trained your first model. Check the log directory for accuracy metrics and results.

---

## Advanced Usage Guide

Now that you've got your first model running, here's what else you can do with PrismRCL:

### Working with Different Data Types
PrismRCL supports:
- Text classification (as shown above)
- Image classification (PNG format)
- Tabular data (space-separated values)

### Evaluation Methods
Choose based on your data type:
- Text: `naivebayes` (required)
- Images: `fractal` or `chisquared`
- Tabular: `chisquared` (recommended)

### Example Datasets
Text:
- [IMDB Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)
- [AG News](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)

Images:
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

Tabular:
- [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- [Wisconsin Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

### Advanced Features
- [Auto-Optimization](#auto-optimization)
- [Transfer Learning](#transfer-learning)
- [Web-Based Inference](#web-based-inference)
- [Model Management](#model-management)

## Auto-Optimization

Once you're comfortable with basic model training, you can use auto-optimize to improve performance:

```bash
PrismRCL.exe auto-optimize data="C:\PrismRCL\data\train-data" log="C:\PrismRCL\log_files\"
```

PrismRCL automatically selects optimization goals based on your dataset:
- Two classes → Overall Accuracy (acc)
- Balanced dataset → Macro Average F1 Score (af1)
- Imbalanced dataset → Weighted Average F1 Score (wf1)

You can also specify a goal:
```bash
PrismRCL.exe auto-optimize=mcc data="C:\PrismRCL\data\train-data" log="C:\PrismRCL\log_files\"
```

## Working with Images

For image classification, use this command structure:

```bash
PrismRCL.exe fractal rclicks=15 data="C:\PrismRCL\data\image-dataset\train-data" testsize=0.1 savemodel="C:\PrismRCL\models\image-classifier.classify" log="C:\PrismRCL\logfiles\job_folder" stopwhendone
```

Key differences from text classification:
- Use `fractal` instead of `naivebayes`
- Set `rclicks` between 15-20 for images
- Remove `llm` and `directional` parameters

## Working with Tabular Data

For tabular data, each sample should be space-separated values in a single line. Use this command:

```bash
PrismRCL.exe chisquared data="C:\PrismRCL\data\tabular-dataset\train-data" testsize=0.2 savemodel="C:\PrismRCL\models\tabular-classifier.classify" log="C:\PrismRCL\logfiles\job_folder" stopwhendone
```

Note: Since version 2.4.0, you don't need to normalize tabular data values.

## Transfer Learning

To build upon a pre-trained model:

```bash
PrismRCL.exe naivebayes rclicks=15 transferlearn="C:\PrismRCL\models\base_model.classify" data="C:\PrismRCL\data\dataset\train-data" testsize=0.1 savemodel="C:\PrismRCL\models\enhanced_model.classify" log="C:\PrismRCL\logfiles\job_folder" stopwhendone
```

## Model Management

### Combining Models
You can combine models trained on similar data:

```bash
PrismRCL.exe loadmodel="C:\PrismRCL\models\model1.classify" addmodel="C:\PrismRCL\models\model2.classify;C:\PrismRCL\models\model3.classify" savemodel="C:\PrismRCL\models\combined_model.classify" stopwhendone
```

## Web-Based Inference

For production deployment, use PrismRCLM.exe to create a web inference endpoint:

```bash
PrismRCLM.exe loadmodel="C:\models\model111.classify" port=8080
```

Access your model via HTTP:
```
http://localhost:8080/evaluation_method&imaginary_flag&testdata=C:/RCLC/data/test_data/&log=C:/RCLC/logfiles/&inftotext=C:/RCLC/output/prediction_output.txt
```

Server responses:
- `ready`: Server idle
- `done`: Request processed
- `busy`: Processing another request
- `error`: Request error

## Enterprise Features

Enterprise users get access to:
- Docker container support
- Multiple model loading on different ports
- Network drive support
- Advanced logging and monitoring
- Automated deployment tools

## Version Notes

- Version 2.6.0 models are not backward compatible
- Earlier version models won't work with 2.6.0
- Docker support requires Enterprise license

## Tips for Success

1. Name models descriptively: `dataset-type_v260_date.classify`
2. Match evaluation methods to data:
   - Images → `fractal` or `chisquared`
   - Text → `naivebayes`
   - Tabular → `chisquared`
3. Start with auto-optimize for new datasets
4. Check log files for detailed metrics
5. Use `stopwhendone` for automated workflows

## Getting Help

- Check log files for detailed error messages
- Contact support for technical assistance
- Visit documentation portal for updates
- Enterprise users: Contact your account manager for deployment help
