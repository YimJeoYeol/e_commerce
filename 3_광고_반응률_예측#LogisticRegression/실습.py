import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("advertising.csv")

df.tail(10)

df.describe()

df.info()

sns.displot(df["Age"])
plt.show()


df["Country"]