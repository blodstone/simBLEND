import logging
from argparse import ArgumentParser
from math import log
from pathlib import Path
import re
import select

from omegaconf import OmegaConf
from transformers.models.auto.tokenization_auto import AutoTokenizer
from data_modules.indices_data import SeqVQVAEDataModule
from data_modules.mind_recsys_data import MINDRecSysDataModule
from data_modules.vq_vae_data import VQVAEDataModule
from modules.aspect_enc import AspectRepr
from modules.llama_decoder import LlamaDecoderForNextArticle
from modules.lstur import LSTUR
from modules.res_vqvae import RVQVAE
import pandas as pd
from typing import List, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def setup_decoder_model(config_path):
    logging.info(f"Loading decoder model from config: {config_path}")
    conf = OmegaConf.load(config_path)
    seqvqvae = LlamaDecoderForNextArticle.load_from_checkpoint(
        conf.checkpoint_path,
        codebook_size=conf.codebook_size,
        hidden_size=conf.hidden_size,
        intermediate_size=conf.intermediate_size,
        num_hidden_layers=conf.num_hidden_layers,
        num_attention_heads=conf.num_attention_heads,
        max_position_embeddings=conf.max_position_embeddings)
    seqvqvae.eval()
    logging.info("Finished loading decoder model.")
    return seqvqvae

def load_grouped_articles(file_path):
    # Placeholder for loading grouped articles
    columns = ['date', 'articles']
    df = pd.read_csv(file_path, sep='\t', header=None, names=columns)
    return df

def load_news_data(file_path):
    columns_names = [
                "nid",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
            ]
    news_df = pd.read_csv(file_path, sep='\t', header=None, names=columns_names)
    return news_df

def load_behavior_data(file_path):
    columns = ['impression_id', 'user_id', 'timestamp', 'history', 'impressions']
    behavior_df = pd.read_csv(file_path, sep='\t', header=None, names=columns)
    return behavior_df

def encode_raw_article(article, aspect_model, tokenizer):
    inputs = tokenizer(article, return_tensors="pt", padding=True, truncation=True)
    return aspect_model.encode_article(**inputs)

def generate_impressions(
    history: List[str],
    tokenizer: Any,
    decoder_model: Any,
    vqvae_models: List[Any],
    aspect_models: List[Any],
    selected_aspects: List[str]
):
    aspect = selected_aspects[0]  # Placeholder for aspect selection logic
    logging.info(f"Generating impressions for aspect: {aspect}")

def main(args):
    """
    Main function for generating article impressions. 
    The input is a set of user history and annotated impressions.
    The decoder model generates candidate articles based on the user's history.
    """

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    decoder_model = setup_decoder_model(args.decoder_config_path)
    article_groups_df = load_grouped_articles(args.grouped_articles_path)
    news_df = load_news_data(args.news_path)
    behavior_df = load_behavior_data(args.behavior_path)
    logging.info("Starting candidate generation...")
    logging.info('Encode article groups using aspect representation and insert into the database')
    for index, row in article_groups_df.iterrows():
        date = row['date']
        articles = row['articles'].split()

        logging.info(f"Processing date {date} with articles {articles}")
        # Here you would encode the articles using the aspect model and insert into the database
        # This is a placeholder for the actual encoding logic
    
    for index, row in behavior_df.iterrows():
        user_id = row['user_id']
        history = row['history'].split()
        impressions = row['impressions']
        predicted_tokens = predict_tokens(history, decoder_model)
        gen_impressions = generate_impressions(history, tokenizer, decoder_model, vqvae_models, selected_aspects)
        # Process the history and impressions to generate candidates
        # This is a placeholder for the actual candidate generation logic
        logging.info(f"Processing user {user_id} with history {history} and impressions {impressions}")
        # Here you would use the decoder_model, vqvae_models, and recsys_model to generate candidates

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--decoder_config_path', type=str, required=True,
                        help='Path to the decoder model config')
    parser.add_argument('--recsys_config_path', type=str, required=True, help='Paths to the Recommender System model config')
    parser.add_argument('--aspects_dictionary_paths', type=str, nargs='+', required=True,
                        help='Paths to the aspect dictionaries')
    parser.add_argument('--codebook_dictionary_paths', type=str, nargs='+', required=True,
                        help='Paths to the codebook dictionaries')
    parser.add_argument('--grouped_articles_path', type=str, required=True,
                        help='Path to the grouped articles')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to the output file')
    parser.add_argument('--news_path', type=str, required=True,
                        help='Path to the news data')
    parser.add_argument('--behavior_path', type=str, required=True,
                        help='Path to the behavior data')
    args = parser.parse_args()
    main(args)