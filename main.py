from src.data_loader import load_both_datasets
from src.preprocessing import preprocess_dataset,merge, rolling_mean, apply_smote
from src.visualization import visualize_data
import pandas as pd
from sklearn.model_selection import train_test_split

from src.train_models import train_and_evaluate


pd.set_option('display.max_columns', None)  # Show all columns
   

df1, df2, df3 = load_both_datasets()
df1_clean = preprocess_dataset(df1)

df2_clean = preprocess_dataset(df2)

df = merge(df1_clean, df2_clean)

df3_clean = preprocess_dataset(df3)

#visualize_data(df3_clean)
#print(df3_clean.shape)
train_and_evaluate(df3_clean)

