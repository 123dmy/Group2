import scipy.stats as stats
import pandas as pd

def conduct_chi_square(dataset):
    if dataset is not None:
        # Step 1: Display variables
        print("\nFor Chi-Square test, following are the variables available:")
        print(f"{'Variable':<20}{'Type':<15}")
        print("-" * 35)

        categorical_vars = []

        for column in dataset.columns:
            if dataset[column].dtype == 'object' or dataset[column].nunique() <= 10:
                print(f"{column:<20}{'Categorical':<15}")
                categorical_vars.append(column)

        # Step 2: Get two categorical variables from user
        if len(categorical_vars) >= 2:
            while True:
                var1 = input("Enter the first categorical variable: ")
                if var1 in categorical_vars:
                    break
                print("Invalid choice. Please select a valid categorical variable.")

            while True:
                var2 = input("Enter the second categorical variable: ")
                if var2 in categorical_vars:
                    break
                print("Invalid choice. Please select a valid categorical variable.")

            print(f"\nPerforming Chi-Square test over the selected variables: {var1} and {var2}…")

            # Step 3: Create a contingency table
            contingency_table = pd.crosstab(dataset[var1], dataset[var2])

            # Step 4: Perform Chi-Square test
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

            print(f"Chi-Square Test Result:\nChi-Square Statistic: {chi2_stat:.6f}\np-value: {p_value:.6f}\nDegrees of Freedom: {dof}")
            print("Expected frequencies:\n", expected)

            if p_value < 0.05:
                print("Result is statistically significant.")
                print(f"There is a statistically significant association between ‘{var1}’ and ‘{var2}’.")
            else:
                print("Result is not statistically significant.")
        else:
            print("Not enough categorical variables found for Chi-Square test.")
    else:
        print("Dataset is not available for analysis.")




