#!/usr/bin/env python3
"""
Preprocessor script for hate speech detection datasets.
Currently handles the Ayele et al. (2023) dataset.
"""

import pandas as pd
import os
import re
from typing import List, Dict, Tuple, Optional
import logging
import numpy as np
import csv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HateSpeechPreprocessor:
    """Preprocessor for hate speech detection datasets."""
    
    def __init__(self, data_dir: str = "dataset"):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Directory containing the datasets
        """
        self.data_dir = data_dir
        
    def parse_ayele_dataset(self, file_path: str) -> List[Dict[str, str]]:
        """
        Parse Ayele et al. (2023) dataset format.
        
        The format is: __label__{label} {text}
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of dictionaries with 'text' and 'label' keys
        """
        data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse the label and text
                    # Format: __label__{label} {text}
                    match = re.match(r'__label__(\w+)\s+(.+)', line)
                    if match:
                        label = match.group(1)
                        text = match.group(2).strip()
                        # Merge 'offensive' into 'hate'
                        if label == 'offensive':
                            label = 'hate'
                        data.append({
                            'text': text,
                            'label': label,
                            'source_file': os.path.basename(file_path)
                        })
                    else:
                        logger.warning(f"Could not parse line {line_num} in {file_path}: {line[:100]}...")
                        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []
            
        return data
    
    def combine_ayele_dataset(self) -> pd.DataFrame:
        """
        Combine all Ayele et al. (2023) dataset files into a single DataFrame.
        
        Returns:
            Combined DataFrame with columns: text, label, split, dataset
        """
        ayele_dir = os.path.join(self.data_dir, "RANLP2023")
        
        if not os.path.exists(ayele_dir):
            raise FileNotFoundError(f"RANLP2023 dataset directory not found: {ayele_dir}")
        
        all_data = []
        
        # Process each split
        splits = {
            'train.csv': 'train',
            'dev.csv': 'dev', 
            'test.csv': 'test'
        }
        
        for filename, split_name in splits.items():
            file_path = os.path.join(ayele_dir, filename)
            
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            logger.info(f"Processing {filename}...")
            data = self.parse_ayele_dataset(file_path)
            
            # Add split information
            for item in data:
                item['split'] = split_name
                # Remove source_file if present
                if 'source_file' in item:
                    del item['source_file']
                
            all_data.extend(data)
            logger.info(f"Loaded {len(data)} samples from {filename}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Add dataset information
        df['dataset'] = 'RANLP2023'
        
        logger.info(f"Combined dataset shape: {df.shape}")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        logger.info(f"Split distribution:\n{df['split'].value_counts()}")
        
        return df
    
    def save_combined_dataset(self, df: pd.DataFrame, output_path: Optional[str] = None):
        """
        Save the combined dataset to a CSV file.
        
        Args:
            df: Combined DataFrame
            output_path: Output file path (if None, defaults to dataset/combined_dataset.csv)
        """
        try:
            if output_path is None:
                output_path = os.path.join(self.data_dir, "combined_dataset.csv")
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Combined dataset saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
    
    def get_dataset_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            df: Combined DataFrame
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_samples': len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'avg_text_length': df['text'].str.len().mean(),
            'min_text_length': df['text'].str.len().min(),
            'max_text_length': df['text'].str.len().max(),
            'unique_labels': df['label'].nunique(),
            'unique_datasets': df['dataset'].nunique()
        }
        
        return stats

    def parse_md2023_dataset(self, file_path: str) -> list:
        """
        Parse Degu, Mekuanent (2022) dataset (MD2023), normalizing labels to match RANLP2023.
        Args:
            file_path: Path to the Excel file
        Returns:
            List of dicts with 'text', 'label', 'split', 'dataset'
        """
        label_map = {
            'hate': 'hate',
            'free': 'normal',
            'free ': 'normal',
            'hae': 'hate',
            'ree': 'normal',
        }
        data = []
        try:
            df = pd.read_excel(file_path)
            for _, row in df.iterrows():
                label = str(row['label']).strip()
                text = str(row['posts']).strip()
                norm_label = label_map.get(label, None)
                if norm_label and text:
                    data.append({
                        'text': text,
                        'label': norm_label,
                        'split': 'all',
                        'dataset': 'MD2023'
                    })
        except Exception as e:
            logger.error(f"Error reading MD2023 file {file_path}: {e}")
        return data

    def parse_sm2022_dataset(self, file_path: str) -> list:
        """
        Parse SM2022 (Gashe et al. 2022) dataset, normalizing labels to 'normal' (2) and 'hate' (others).
        Args:
            file_path: Path to the CSV file
        Returns:
            List of dicts with 'text', 'label', 'dataset'
        """
        data = []
        try:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                label = str(row['label']).strip()
                text = str(row['PostContent']).strip()
                if not text:
                    continue
                if label == '2':
                    norm_label = 'normal'
                else:
                    norm_label = 'hate'
                data.append({
                    'text': text,
                    'label': norm_label,
                    'dataset': 'SM2022'
                })
        except Exception as e:
            logger.error(f"Error reading SM2022 file {file_path}: {e}")
        return data

    def parse_zak2021_dataset(self, train_path: str, test_path: str) -> list:
        """
        Parse ZAK2021 (Kassa, Zeleke Abebaw 2021) dataset, normalizing labels to 'hate' and 'normal'.
        Args:
            train_path: Path to the train txt file
            test_path: Path to the test txt file
        Returns:
            List of dicts with 'text', 'label', 'dataset'
        """
        data = []
        label_map = {
            'ጥላቻ': 'hate',
            'መልካም': 'normal',
        }
        for file_path in [train_path, test_path]:
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    text = str(row['Content']).strip()
                    label = str(row['Label']).strip()
                    if not text or not label or label == 'nan':
                        continue
                    norm_label = label_map.get(label, None)
                    if norm_label:
                        data.append({
                            'text': text,
                            'label': norm_label,
                            'dataset': 'ZAK2021'
                        })
            except Exception as e:
                logger.error(f"Error reading ZAK2021 file {file_path}: {e}")
        return data

    def parse_sg2020_dataset(self, posts_path: str, labels_path: str) -> list:
        """
        Parse SG2020 (Tesfaye and Kakeba 2020) dataset, normalizing labels to 'hate' and 'normal'.
        Args:
            posts_path: Path to the posts.txt file
            labels_path: Path to the labels.txt file
        Returns:
            List of dicts with 'text', 'label', 'dataset'
        """
        data = []
        try:
            with open(posts_path, 'r', encoding='utf-8') as pf, open(labels_path, 'r', encoding='utf-8') as lf:
                for post, label in zip(pf, lf):
                    text = post.strip()
                    label = label.strip().replace(' ', '')
                    if not text or not label:
                        continue
                    if label.lower() == 'free':
                        norm_label = 'normal'
                    else:
                        norm_label = 'hate'
                    data.append({
                        'text': text,
                        'label': norm_label,
                        'dataset': 'SG2020'
                    })
        except Exception as e:
            logger.error(f"Error reading SG2020 files {posts_path}, {labels_path}: {e}")
        return data

    def parse_trac_hm2024_dataset(self, file_path: str) -> list:
        """
        Parse TRAC-HM2024 dataset, normalizing labels to 'hate' and 'normal'.
        Args:
            file_path: Path to the Excel file
        Returns:
            List of dicts with 'text', 'label', 'dataset'
        """
        data = []
        label_map = {
            'normal speech': 'normal',
            'hate speech': 'hate',
        }
        try:
            df = pd.read_excel(file_path)
            for _, row in df.iterrows():
                text = str(row['text']).strip()
                label = str(row['Lable']).strip().lower()
                norm_label = label_map.get(label, None)
                if norm_label and text:
                    data.append({
                        'text': text,
                        'label': norm_label,
                        'dataset': 'TRAC-HM2024'
                    })
        except Exception as e:
            logger.error(f"Error reading TRAC-HM2024 file {file_path}: {e}")
        return data

    def parse_trac_hi2024_dataset(self, file_paths: list) -> list:
        """
        Parse TRAC-HI2024 dataset, normalizing labels to 'hate' and 'normal'.
        Args:
            file_paths: List of CSV file paths
        Returns:
            List of dicts with 'text', 'label', 'dataset'
        """
        data = []
        label_map = {
            'normal': 'normal',
            'hate': 'hate',
            'offensive': 'hate',
        }
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    text = str(row['tweet']).strip()
                    label = str(row['label']).strip().lower()
                    norm_label = label_map.get(label, None)
                    if norm_label and text:
                        data.append({
                            'text': text,
                            'label': norm_label,
                            'dataset': 'TRAC-HI2024'
                        })
            except Exception as e:
                logger.error(f"Error reading TRAC-HI2024 file {file_path}: {e}")
        return data

    def combine_all_datasets(self) -> pd.DataFrame:
        """
        Combine RANLP2023, MD2023, SM2022, ZAK2021, SG2020, TRAC-HM2024, and TRAC-HI2024 datasets into a single DataFrame.
        Returns:
            Combined DataFrame with columns: text, label, dataset
        """
        # RANLP2023
        ranlp_df = self.combine_ayele_dataset()
        # MD2023
        md2023_path = os.path.join(self.data_dir, "MD2023", "D1_org.xlsx")
        md2023_data = self.parse_md2023_dataset(md2023_path)
        md2023_df = pd.DataFrame(md2023_data)
        # SM2022
        sm2022_path = os.path.join(self.data_dir, "SM2022", "normalized_data.csv")
        sm2022_data = self.parse_sm2022_dataset(sm2022_path)
        sm2022_df = pd.DataFrame(sm2022_data)
        # ZAK2021
        zak2021_train = os.path.join(self.data_dir, "ZAK2021", "AMHSDataTrain.txt")
        zak2021_test = os.path.join(self.data_dir, "ZAK2021", "AMHSDataTest.txt")
        zak2021_data = self.parse_zak2021_dataset(zak2021_train, zak2021_test)
        zak2021_df = pd.DataFrame(zak2021_data)
        # SG2020
        sg2020_posts = os.path.join(self.data_dir, "SG2020", "posts.txt")
        sg2020_labels = os.path.join(self.data_dir, "SG2020", "labels.txt")
        sg2020_data = self.parse_sg2020_dataset(sg2020_posts, sg2020_labels)
        sg2020_df = pd.DataFrame(sg2020_data)
        # TRAC-HM2024
        trac_hm2024_path = os.path.join(self.data_dir, "TRAC-HM2024", "Preprocessed_dataset.xlsx")
        trac_hm2024_data = self.parse_trac_hm2024_dataset(trac_hm2024_path)
        trac_hm2024_df = pd.DataFrame(trac_hm2024_data)
        # TRAC-HI2024
        trac_hi2024_files = [
            os.path.join(self.data_dir, "TRAC-HI2024", "train_category.csv"),
            os.path.join(self.data_dir, "TRAC-HI2024", "dev_category.csv"),
            os.path.join(self.data_dir, "TRAC-HI2024", "test_category.csv"),
        ]
        trac_hi2024_data = self.parse_trac_hi2024_dataset(trac_hi2024_files)
        trac_hi2024_df = pd.DataFrame(trac_hi2024_data)
        # Combine
        combined_df = pd.concat([
            ranlp_df, md2023_df, sm2022_df, zak2021_df, sg2020_df, trac_hm2024_df, trac_hi2024_df
        ], ignore_index=True)
        # Remove split column if present
        if 'split' in combined_df.columns:
            combined_df = combined_df.drop(columns=['split'])
        logger.info(f"Combined all datasets shape: {combined_df.shape}")
        logger.info(f"All datasets label distribution:\n{combined_df['label'].value_counts()}")
        logger.info(f"All datasets source distribution:\n{combined_df['dataset'].value_counts()}")
        return combined_df

def main():
    """Main function to run the preprocessor."""
    logger.info("Starting hate speech dataset preprocessing...")
    preprocessor = HateSpeechPreprocessor()
    try:
        # Combine all datasets
        logger.info("Processing all datasets (RANLP2023 + MD2023 + SM2022 + ZAK2021 + SG2020 + TRAC-HM2024 + TRAC-HI2024)...")
        combined_df = preprocessor.combine_all_datasets()
        # Get and display statistics
        stats = preprocessor.get_dataset_stats(combined_df)
        logger.info("Dataset Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        # Save combined dataset
        preprocessor.save_combined_dataset(combined_df)
        logger.info("Preprocessing completed successfully!")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    main() 