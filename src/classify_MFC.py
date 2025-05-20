from argparse import ArgumentParser
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import torch.nn.functional as F
from data_modules.mind_component import load_news_data
import logging
from tqdm import tqdm

def classify_news(texts, tokenizer, model, config, batch_size=32):
    device = model.device
    all_scores = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        encoded_input = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        logits = model(**encoded_input.to(device)).logits
        scores = F.sigmoid(logits).detach().cpu().numpy()
        assert len(scores) == len(batch_texts)
        all_scores.extend(scores)

    # After processing all batches, determine predicted labels
    predicted_labels = []
    for scores in all_scores:
        max_score_index = scores.argmax()
        predicted_labels.append(config.id2label[max_score_index])
        
    if not hasattr(config, 'id2label') or not config.id2label:
        raise ValueError("The model configuration does not contain 'id2label'. Please ensure the configuration includes label mappings.")
    predicted_labels = [config.id2label[score_set.argmax()] for score_set in all_scores]
    return predicted_labels


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

    tokenizer = AutoTokenizer.from_pretrained("sourabhdattawad/mfc-xlm-roberta")
    config = AutoConfig.from_pretrained("sourabhdattawad/mfc-xlm-roberta")
    config.id2label = {i: label for i, label in enumerate(config.label2id.keys())}  # Ensure id2label is set
    model = AutoModelForSequenceClassification.from_pretrained("sourabhdattawad/mfc-xlm-roberta")
    model.to(device)
    model.eval()
    log.info("Classifying training data")
    df_train_news['frame'] = classify_news(df_train_news['text'].tolist(), tokenizer, model, config)
    output_train_dir = Path(args.train_path).parent / f'MINDlarge_{args.output_name}_train'
    output_train_dir.mkdir(parents=True, exist_ok=True)
    df_train_news.to_csv(output_train_dir / f'news.tsv', index=False, header=False, sep='\t')
    log.info("Classifying dev data")
    df_dev_news['frame'] = classify_news(df_dev_news['text'].tolist(), tokenizer, model, config)
    output_dev_dir = Path(args.dev_path).parent / f'MINDlarge_{args.output_name}_dev'
    output_dev_dir.mkdir(parents=True, exist_ok=True)
    df_dev_news.to_csv(output_dev_dir / f'news.tsv', index=False, header=False, sep='\t')
    log.info("Classifying test data")
    df_test_news['frame'] = classify_news(df_test_news['text'].tolist(), tokenizer, model, config)
    output_test_dir = Path(args.test_path).parent / f'MINDlarge_{args.output_name}_test'
    output_test_dir.mkdir(parents=True, exist_ok=True)
    df_test_news.to_csv(output_test_dir / f'news.tsv', index=False, header=False, sep='\t')
    

if __name__ == '__main__':
    parser = ArgumentParser(description="Classify article into its MFC topic")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to the development dataset.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument('--output_name', type=str, required=True, help='Name of the output file.')
    args = parser.parse_args()
    main(args)