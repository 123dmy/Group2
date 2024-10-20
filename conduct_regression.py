import statsmodels.api as sm
import pandas as pd

def conduct_regression(dataset):
    if dataset is not None:
        # Step 1: Display variables
        print("\nFor Regression analysis, following are the variables available:")
        print(f"{'Variable':<20}{'Type':<15}")
        print("-" * 35)

        continuous_vars = []

        for column in dataset.columns:
            if dataset[column].dtype != 'object':
                print(f"{column:<20}{'Continuous':<15}")
                continuous_vars.append(column)

        # Step 2: Get dependent and independent variables from user
        if len(continuous_vars) >= 2:
            while True:
                dep_var = input("Enter the dependent (response) variable: ")
                if dep_var in continuous_vars:
                    break
                print("Invalid choice. Please select a valid continuous variable.")

            print("Available independent variables:")
            for var in continuous_vars:
                if var != dep_var:
                    print(f"- {var}")

            while True:
                indep_var = input("Enter the independent (predictor) variable: ")
                if indep_var in continuous_vars and indep_var != dep_var:
                    break
                print("Invalid choice. Please select a valid continuous variable different from the dependent variable.")

            print(f"\nPerforming Linear Regression with '{dep_var}' as dependent and '{indep_var}' as independent variable…")

            # Step 3: Prepare the data for regression
            X = dataset[indep_var]
            y = dataset[dep_var]
            X = sm.add_constant(X)  # Adds a constant term to the predictor

            # Step 4: Fit the regression model
            model = sm.OLS(y, X).fit()

            # Step 5: Print the regression results
            print(model.summary())
        else:
            print("Not enough continuous variables found for Regression analysis.")
    else:
        print("Dataset is not available for analysis.")

