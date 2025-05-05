
from pathlib import Path
import argparse
import torch
import logging
import tqdm
import numpy as np
from data_modules.mind_aspect_data import MINDEncDataModule, NewsBatch
from modules.aspect_enc import AspectRepr

def save_vector_dict(vector_dict, output_path):
    """
    Saves a dictionary of vectors to a file, with each line containing
    the key (index/ID) followed by the vector components.

    Args:
        vector_dict (dict): A dictionary where keys are indices/IDs (strings or numbers)
                             and values are NumPy arrays or lists representing the vectors.
        output_path (str): The path to the output file.
    """    
    output_dir = Path(output_path).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for key, vector in vector_dict.items():
            # Ensure the vector is a NumPy array and flatten it if necessary
            vector_array = np.array(vector).flatten()
            # Convert vector elements to strings and join them with spaces
            vector_str = ' '.join(map(str, vector_array))
            # Write the key and the vector string to the file
            f.write(f"{key} {vector_str}\n")
    
def process_dataloader(logger, device, model, dataloader, output_file):
    logger.info(f'Processing dataloader of {output_file}...')
    save_dataset = {}
    for article in tqdm.tqdm(dataloader):
        batch = NewsBatch(article)
        batch["news"]["text"] = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch["news"]["text"].items()}
        batch['labels'] = batch['labels'].to(device)  # Move labels to the same device
        nid = batch['news']['news_ids']
        with torch.no_grad():
            # Forward pass to get embeddings
            try:
                representation = model.forward(batch)
                save_dataset[nid[0]] = representation.cpu().numpy()
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue
    save_vector_dict(
        vector_dict=save_dataset,
        output_path=Path(args.output_folder) / output_file
    )
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save aspect representation vectors.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the aspect representation model checkpoint.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the output vector files.")
    parser.add_argument('--output_name', type=str, required=True, help='Name of the output file.')
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to the development dataset.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting vector saving process...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AspectRepr.load_from_checkpoint(args.model_path) 
    MIND_dev_path = Path(args.dev_path)
    MIND_train_path = Path(args.train_path)
    MIND_test_path = Path(args.test_path)
    mind = MINDEncDataModule(train_path=MIND_train_path, dev_path=MIND_dev_path, test_path=MIND_test_path, batch_size=1)
    mind.setup("fit") 
    process_dataloader(logger, device, model, mind.train_dataloader(shuffle=False, num_workers=1), f'train_{args.output_name}_aspect_vectors.txt')
    process_dataloader(logger, device, model, mind.val_dataloader(), f'dev_{args.output_name}_aspect_vectors.txt')
    mind.setup("test") 
    process_dataloader(logger, device, model, mind.test_dataloader(), f'test_{args.output_name}_aspect_vectors.txt')
    logger.info("Vector saving process completed.")

    
