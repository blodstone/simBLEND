from argparse import ArgumentParser
from pathlib import Path
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import torch.nn.functional as F
from data_modules.mind_component import load_news_data
import logging
from tqdm import tqdm

def classify_news(texts, model, tokenizer, batch_size=64):
    all_labels = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
        labels = [sentiment_map[p] for p in torch.argmax(probabilities, dim=-1).tolist()]
        all_labels.extend(labels)
    return all_labels


def main(args):
    """
    Main function to classify news articles into their respective MFC topics.

    Args:
        args: Parsed command-line arguments containing paths to the train, dev, and test datasets,
              as well as the output file name.
    """
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    log.info("Loading training data from %s", args.train_path)
    df_train_news = load_news_data(Path(args.train_path), 'train')
    df_dev_news = load_news_data(Path(args.dev_path), 'dev')
    df_test_news = load_news_data(Path(args.test_path), 'test')
    model_name = "tabularisai/multilingual-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Classify a new sentence
    log.info("Classifying training data")
    df_train_news['sentiment'] = classify_news(df_train_news['text'].tolist(), model, tokenizer)
    output_train_dir = Path(args.train_path).parent / f'MINDlarge_{args.output_name}_train'
    output_train_dir.mkdir(parents=True, exist_ok=True)
    df_train_news.to_csv(output_train_dir / f'news.tsv', index=False, header=False, sep='\t')
    log.info("Classifying dev data")
    df_dev_news['sentiment'] = classify_news(df_dev_news['text'].tolist(), model, tokenizer)
    output_dev_dir = Path(args.dev_path).parent / f'MINDlarge_{args.output_name}_dev'
    output_dev_dir.mkdir(parents=True, exist_ok=True)
    df_dev_news.to_csv(output_dev_dir / f'news.tsv', index=False, header=False, sep='\t')
    log.info("Classifying test data")
    df_test_news['sentiment'] = classify_news(df_test_news['text'].tolist(), model, tokenizer)
    output_test_dir = Path(args.test_path).parent / f'MINDlarge_{args.output_name}_test'
    output_test_dir.mkdir(parents=True, exist_ok=True)
    df_test_news.to_csv(output_test_dir / f'news.tsv', index=False, header=False, sep='\t')
    

if __name__ == '__main__':
    parser = ArgumentParser(description="Classify article into its sentiment")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to the development dataset.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument('--output_name', type=str, required=True, help='Name of the output file.')
    args = parser.parse_args()
    main(args)