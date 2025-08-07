#!/usr/bin/env python3
"""
Script to regenerate PCA visualizations with result day coloring.

This script will run the PCA visualization code in the notebooks to ensure
all plots use the new result day coloring scheme.
"""

import sys
import os
sys.path.append('..')

from src import (
    Config, 
    load_model_and_tokenizer,
    TemporalDatasetCreator,
    analyze_all_layers,
    create_layer_pca_visualization
)
import matplotlib.pyplot as plt

def main():
    # Initialize config
    config = Config()
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Create datasets
    print("Creating datasets...")
    dataset_creator = TemporalDatasetCreator(config)
    dataset = dataset_creator.create_dataset(['add_sing', 'add_plur'])
    
    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(dataset)
    
    # Separate by context
    add_sing_df = df[df['context'] == 'add_sing'].reset_index(drop=True)
    add_plur_df = df[df['context'] == 'add_plur'].reset_index(drop=True)
    
    # Split into train/test (using all as train for simplicity)
    train_sing = add_sing_df
    test_sing = add_sing_df.iloc[:0]  # Empty test set
    train_plur = add_plur_df  
    test_plur = add_plur_df.iloc[:0]  # Empty test set
    
    # Analyze layers
    print("Analyzing add_sing...")
    sing_results = analyze_all_layers(
        train_sing, test_sing, 'add_sing', 
        model=model, tokenizer=tokenizer, config=config
    )
    
    print("Analyzing add_plur...")  
    plur_results = analyze_all_layers(
        train_plur, test_plur, 'add_plur',
        model=model, tokenizer=tokenizer, config=config
    )
    
    # Create visualizations with new coloring
    print("\nCreating PCA visualizations with RESULT DAY coloring...")
    
    # add_sing visualization
    fig_sing = create_layer_pca_visualization(sing_results, 'add_sing', color_by_result=True)
    sing_save_path = "plots/pca_projections/pca_all_layers_add_sing_result_colored.png"
    os.makedirs(os.path.dirname(sing_save_path), exist_ok=True)
    plt.savefig(sing_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {sing_save_path}")
    plt.close()
    
    # add_plur visualization  
    fig_plur = create_layer_pca_visualization(plur_results, 'add_plur', color_by_result=True)
    plur_save_path = "plots/pca_projections/pca_all_layers_add_plur_result_colored.png"
    plt.savefig(plur_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plur_save_path}")
    plt.close()
    
    print("\nDone! PCA visualizations have been updated with result day coloring.")
    print("The new plots show points colored by the RESULT day (the answer), not the starting day.")
    print("For example: 'Saturday + 1 day = Sunday' is now colored as Sunday (the result).")

if __name__ == "__main__":
    main()