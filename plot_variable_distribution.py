import pandas as pd
from matplotlib import pyplot as plt


def plot_distribution(self, variable):
    if variable in self.dataset.columns:
        if pd.api.types.is_numeric_dtype(self.dataset[variable]):
            self.dataset[variable].hist()
            plt.title(f"Distribution plot for '{variable}'")  # 更新图表标题
            plt.xlabel(variable)
            plt.ylabel("Frequency")
            plt.show()
        else:
            print(f"Variable '{variable}' is not numerical, can't plot distribution.")
    else:
        print(f"Variable '{variable}' does not exist in the dataset.")
