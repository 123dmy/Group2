import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def conduct_t_test(dataset):
    if dataset is not None:
        # Step 1: Display variables
        print("\nFor t-Test, following are the variables available:")
        print(f"{'Variable':<20}{'Type':<15}")
        print("-" * 35)

        categorical_vars = []
        continuous_vars = []

        for column in dataset.columns:
            dtype = str(dataset[column].dtype)
            if dtype == 'object' or dataset[column].nunique() <= 10:
                var_type = 'Categorical'
                categorical_vars.append(column)
            else:
                var_type = 'Continuous'
                continuous_vars.append(column)
            print(f"{column:<20}{var_type:<15}")

        # Step 2: Get continuous and categorical variables from user
        if continuous_vars and categorical_vars:
            while True:
                cont_var = input("Enter a continuous (interval/ratio) variable: ")
                if cont_var in continuous_vars:
                    break
                print("Invalid choice. Please select a valid continuous variable.")

            while True:
                cat_var = input("Enter a categorical (binary) variable: ")
                if cat_var in categorical_vars:
                    # Ensure categorical variable has only two unique values
                    if dataset[cat_var].nunique() == 2:
                        break
                    else:
                        print("Please select a binary categorical variable.")
                else:
                    print("Invalid choice. Please select a valid categorical variable.")

            print(f"\nPerforming t-Test over the selected variables: {cont_var} and {cat_var}…")

            # Step 3: Check normality of the continuous variable for each group
            groups = [dataset[dataset[cat_var] == group][cont_var].dropna() for group in dataset[cat_var].unique()]
            normal_distributed = True
            for group in groups:
                stat, p_value = stats.shapiro(group)
                if p_value < 0.05:
                    print(f"Group with '{cat_var}' value '{group.name}' is not normally distributed.")
                    normal_distributed = False

            if normal_distributed:
                print(f"All groups are normally distributed. Performing t-Test…")
                t_stat, p_value = stats.ttest_ind(*groups, equal_var=True)
                print(f"t-Test Result:\nt-statistic: {t_stat:.6f}\np-value: {p_value:.6f}")

                if p_value < 0.05:
                    print("Result is statistically significant.")
                    print(f"There is a statistically significant difference in the average ‘{cont_var}’ across the categories of ‘{cat_var}’.")
                else:
                    print("Result is not statistically significant.")
            else:
                print("One or more groups are not normally distributed, consider using a non-parametric test instead.")
                print(f"Performing Mann-Whitney U test instead of t-Test…")
                mann_whitney_stat, mann_whitney_p_value = stats.mannwhitneyu(groups[0], groups[1])

                print(f"Mann-Whitney U Test Result:\nU-statistic: {mann_whitney_stat:.6f}\np-value: {mann_whitney_p_value:.6f}")
                if mann_whitney_p_value < 0.05:
                    print("Result is statistically significant.")
                    print(f"There is a statistically significant difference in the distribution of ‘{cont_var}’ between the groups of ‘{cat_var}’.")
                else:
                    print("Result is not statistically significant.")
        else:
            print("No appropriate variables found for t-Test. Ensure the dataset contains both categorical and continuous variables.")
    else:
        print("Dataset is not available for analysis.")
