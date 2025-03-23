import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
n_samples = 100

# Generate features
data = {
    'feature1': np.random.rand(n_samples),
    'feature2': np.random.rand(n_samples),
    'feature3': np.random.rand(n_samples),
    'class': ['class1' if x > 0.5 else 'class2' for x in np.random.rand(n_samples)]
}

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('test_dataset.csv', index=False)
print("Test dataset created successfully: test_dataset.csv") 