import pandas as pd
import numpy as np

def create_sample_dataset():
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'feature1': np.random.rand(n_samples),
        'feature2': np.random.rand(n_samples),
        'feature3': np.random.rand(n_samples),
        'class': ['class1' if x > 0.5 else 'class2' 
                 for x in np.random.rand(n_samples)]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = 'data/sample_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created: {output_path}")

if __name__ == "__main__":
    create_sample_dataset() 