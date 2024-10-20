import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def conduct_anova(dataset):
    if dataset is not None:
        # Step 1: Display variables
        print("\nFor ANOVA, following are the variables available:")
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
                cat_var = input("Enter a categorical (ordinal/nominal) variable: ")
                if cat_var in categorical_vars:
                    break
                print("Invalid choice. Please select a valid categorical variable.")

            print(f"\nPerforming ANOVA over the selected variables: {cont_var} and {cat_var}…")

            # Step 3: Check normality of continuous variable
            stat, p_value = stats.shapiro(dataset[cont_var])
            if p_value < 0.05:
                print(f"‘{cont_var}’ is not normally distributed, as shown in the Q-Q plot…")
                # Plot Q-Q plot
                plt.figure(figsize=(6, 4))
                stats.probplot(dataset[cont_var], dist="norm", plot=plt)
                plt.title(f"Q-Q plot for {cont_var}")
                plt.show()

                print(f"Performing Kruskal-Wallis Test instead of ANOVA…")
                # Step 4: Perform Kruskal-Wallis Test
                groups = [group[cont_var].dropna() for name, group in dataset.groupby(cat_var)]
                kruskal_stat, kruskal_p_value = stats.kruskal(*groups)

                print(
                    f"Kruskal-Wallis Result:\nKruskal-Wallis Statistic: {kruskal_stat:.6f}\np-value: {kruskal_p_value:.6f}")
                if kruskal_p_value < 0.05:
                    print("Result is statistically significant.")
                    print(
                        f"There is a statistically significant difference in the average ‘{cont_var}’ across the categories of ‘{cat_var}’.")
                else:
                    print("Result is not statistically significant.")
            else:
                print(f"‘{cont_var}’ is normally distributed. Performing ANOVA…")

                # Step 5: Perform ANOVA
                model = stats.f_oneway(
                    *[dataset[dataset[cat_var] == group][cont_var] for group in dataset[cat_var].unique()])

                print(f"ANOVA Result:\nF-statistic: {model.statistic:.6f}\np-value: {model.pvalue:.6f}")
                if model.pvalue < 0.05:
                    print("Result is statistically significant.")
                    print(
                        f"There is a statistically significant difference in the average ‘{cont_var}’ across the categories of ‘{cat_var}’.")
                else:
                    print("Result is not statistically significant.")
        else:
            print(
                "No appropriate variables found for ANOVA. Ensure the dataset contains both categorical and continuous variables.")
    else:
        print("Dataset is not available for analysis.")
