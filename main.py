import importlib
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


class DataAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = self.load_dataset()

    def load_dataset(self):
        try:
            data = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully from: {self.file_path}")
            return data
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def summarize_variables(self):
        if self.dataset is not None:
            print("Following are the variables in your dataset:")
            print(f"{'Variable':<20}{'Type':<15}{'Mean/Median/Mode':<25}{'Skewness':<10}")
            print("-" * 70)
            for column in self.dataset.columns:
                dtype = str(self.dataset[column].dtype)
                col_type = 'Categorical' if dtype == 'object' else 'Numerical'

                if col_type == 'Numerical':
                    mean_value = self.dataset[column].mean()
                    median_value = self.dataset[column].median()
                    skewness_value = self.dataset[column].skew()
                    mean_median_mode = f"Mean: {mean_value:.2f}, Median: {median_value:.2f}"
                else:
                    mode_value = self.dataset[column].mode()[0] if not self.dataset[column].mode().empty else 'N/A'
                    skewness_value = "N/A"
                    mean_median_mode = f"Mode: {mode_value}"

                print(f"{column:<20}{col_type:<15}{mean_median_mode:<25}{skewness_value:<10}")
        else:
            print("Dataset is not available for summarization.")

    def display_variable_options(self):
        if self.dataset is not None:
            print("\nFollowing variables are available for plot distribution:")
            for i, column in enumerate(self.dataset.columns, start=1):
                print(f"{i}. {column}")
            print(f"{i + 1}. BACK")
            print(f"{i + 2}. QUIT")
            return len(self.dataset.columns) + 2
        else:
            print("No dataset loaded.")
            return 0

    def plot_distribution(self, variable):
        if variable in self.dataset.columns:
            if pd.api.types.is_numeric_dtype(self.dataset[variable]):
                self.dataset[variable].hist()
                plt.title(f"{variable} Distribution")
                plt.xlabel(variable)
                plt.ylabel("Frequency")
                plt.show()
            else:
                print(f"Variable '{variable}' is not numerical, can't plot distribution.")
        else:
            print(f"Variable '{variable}' does not exist in the dataset.")


def load_module(module_name):
    try:
        # Dynamically import the module using importlib
        module = importlib.import_module(module_name)
        print(f"Module '{module_name}' loaded successfully.")
        return module
    except ImportError as e:
        print(f"Error loading module '{module_name}': {e}")
        return None


def get_file_path():
    while True:
        file_path = input("ENTER THE PATH TO YOUR DATASET (.csv): ")
        if os.path.exists(file_path) and file_path.endswith('.csv'):
            print("File found. Proceeding...")
            return file_path
        else:
            print("File not found or invalid format. Please enter a valid .csv file path.")


def main():
    file_path = get_file_path()
    data_analysis = DataAnalysis(file_path)
    data_analysis.summarize_variables()

    while True:
        print("\nHow do you want to analyze your data?")
        print("1. Plot variable distribution")
        print("2. Conduct ANOVA")
        print("3. Conduct t-Test")
        print("4. Conduct chi-Square")
        print("5. Conduct Regression")
        print("6. Conduct Sentiment Analysis")
        print("7. Quit")
        choice = input("Enter your choice (1-7): ")

        if choice == '1':
            variable_count = data_analysis.display_variable_options()
            if variable_count > 0:
                sub_choice = input("Enter your choice: ")
                if sub_choice.isdigit():
                    sub_choice = int(sub_choice)
                    if 1 <= sub_choice <= len(data_analysis.dataset.columns):
                        variable = data_analysis.dataset.columns[sub_choice - 1]
                        data_analysis.plot_distribution(variable)
                    elif sub_choice == variable_count - 1:
                        print("Going back...")
                    elif sub_choice == variable_count:
                        print("Exiting the program...")
                        sys.exit()
                    else:
                        print("Invalid choice, please select a valid option.")
                else:
                    print("Invalid input, please enter a number.")
        elif choice == '2':
            module = load_module('conduct_anova')
            if module:
                module.conduct_anova(data_analysis.dataset)
        elif choice == '3':
            module = load_module('conduct_t_test')
            if module:
                module.conduct_t_test(data_analysis.dataset)
        elif choice == '4':
            module = load_module('conduct_chi_square')
            if module:
                module.conduct_chi_square(data_analysis.dataset)
        elif choice == '5':
            module = load_module('conduct_regression')
            if module:
                module.conduct_regression(data_analysis.dataset)
        elif choice == '6':
            module = load_module('conduct_sentiment_analysis')
            if module:
                module.conduct_sentiment_analysis(data_analysis.dataset)
        elif choice == '7':
            print("Exiting the program...")
            sys.exit()
        else:
            print("Invalid choice, please enter a number between 1 and 7.")


if __name__ == "__main__":
    main()
