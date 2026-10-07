import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


from sklearn.pipeline import Pipeline


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score



# ==================================================
# Create Polynomial Regression Pipeline
# ==================================================

def create_model(model, degree):

    pipeline = Pipeline(
        [

            (
                "poly",
                PolynomialFeatures(
                    degree=degree,
                    include_bias=False
                )
            ),


            (
                "scale",
                StandardScaler()
            ),


            (
                "regressor",
                model
            )

        ]
    )

    return pipeline




# ==================================================
# Train Model and Generate Prediction
# ==================================================

def train_and_predict(
        train_file,
        test_file,
        output_file,
        result_file,
        plot_file,
        max_degree
):


    print("\n================================")
    print("Training:", train_file)
    print("================================")


    # --------------------------
    # Load Dataset
    # --------------------------

    train = pd.read_csv(train_file)

    test = pd.read_csv(test_file)


    print(train.head())

    print("Shape:", train.shape)



    # --------------------------
    # Split Features and Target
    # --------------------------

    X = train.drop(
        "y",
        axis=1
    )


    y = train["y"]



    # --------------------------
    # K Fold Cross Validation
    # --------------------------

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )



    # --------------------------
    # Hyperparameters
    # --------------------------

    alphas = [
        0.01,
        0.1,
        1
    ]



    # --------------------------
    # Create Models
    # --------------------------

    models = []


    for alpha in alphas:


        models.append(
            (
                "Ridge",
                Ridge(
                    alpha=alpha
                ),
                alpha
            )
        )


        models.append(
            (
                "Lasso",
                Lasso(
                    alpha=alpha,
                    max_iter=10000
                ),
                alpha
            )
        )



    # --------------------------
    # Model Search
    # --------------------------

    results = []


    for name, model, alpha in models:


        for degree in range(1, max_degree + 1):


            print(
                "Testing:",
                name,
                "Alpha:",
                alpha,
                "Degree:",
                degree
            )


            pipeline = create_model(
                model,
                degree
            )



            # MSE

            mse_scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=kf,
                scoring="neg_mean_squared_error"
            )


            mse = -mse_scores.mean()



            # R2

            r2_scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=kf,
                scoring="r2"
            )


            r2 = r2_scores.mean()



            results.append(
                [
                    name,
                    alpha,
                    degree,
                    mse,
                    r2
                ]
            )



    # --------------------------
    # Results Table
    # --------------------------

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Alpha",
            "Degree",
            "MSE",
            "R2"
        ]
    )


    print("\nAll Results:")
    print(results_df)



    results_df.to_csv(
        result_file,
        index=False
    )



    # --------------------------
    # Degree vs MSE Plot
    # Best Alpha for Each Model
    # --------------------------

    plt.figure(figsize=(8,5))


    best_alpha_models = results_df.loc[
        results_df.groupby("Model")["MSE"].idxmin()
    ]


    print("\nBest alpha for each model:")
    print(best_alpha_models)



    for _, row in best_alpha_models.iterrows():

        model_name = row["Model"]

        best_alpha = row["Alpha"]


        temp = results_df[
            (results_df["Model"] == model_name)
            &
            (results_df["Alpha"] == best_alpha)
        ]


        plt.plot(
            temp["Degree"],
            temp["MSE"],
            marker="o",
            label=f"{model_name} (alpha={best_alpha})"
        )



    plt.xlabel(
        "Polynomial Degree"
    )


    plt.ylabel(
        "Cross Validation MSE"
    )


    plt.title(
        "Degree vs MSE (Best Alpha)"
    )


    plt.legend()

    plt.grid(True)


    plt.savefig(
        plot_file
    )


    plt.close()



    # --------------------------
    # Select Best Model
    # --------------------------

    best = results_df.loc[
        results_df["MSE"].idxmin()
    ]


    print("\nBest Model:")
    print(best)



    best_model = best["Model"]

    best_alpha = float(
        best["Alpha"]
    )

    best_degree = int(
        best["Degree"]
    )



    # --------------------------
    # Create Final Regressor
    # --------------------------

    if best_model == "Ridge":

        final_regressor = Ridge(
            alpha=best_alpha
        )


    elif best_model == "Lasso":

        final_regressor = Lasso(
            alpha=best_alpha,
            max_iter=10000
        )



    # --------------------------
    # Final Pipeline
    # --------------------------

    final_pipeline = create_model(
        final_regressor,
        best_degree
    )



    # --------------------------
    # Train Full Dataset
    # --------------------------

    final_pipeline.fit(
        X,
        y
    )



    # --------------------------
    # Final R2 Score
    # --------------------------

    final_r2_scores = cross_val_score(
        final_pipeline,
        X,
        y,
        cv=kf,
        scoring="r2"
    )


    final_r2 = final_r2_scores.mean()


    print(
        "\nFinal CV R2:",
        final_r2
    )



    # --------------------------
    # Prediction
    # --------------------------

    prediction = final_pipeline.predict(
        test
    )


    prediction_df = pd.DataFrame(
        {
            "y": prediction
        }
    )


    prediction_df.to_csv(
        output_file,
        index=False
    )


    print(
        "Prediction saved:",
        output_file
    )


    print(
        prediction_df.head()
    )




# ==================================================
# MAIN PROGRAM
# ==================================================


# ==========================
# VAR 1
# Degree Range: 1 - 10
# ==========================

train_and_predict(

    "../data/BT2024081_train_var1(in).csv",

    "../data/BT2024081_test_var1(in).csv",

    "../predictions/BT2024081_pred_var1.csv",

    "../predictions/model_results_var1.csv",

    "../predictions/degree_vs_mse_var1.png",

    10

)



# ==========================
# VAR 2
# Degree Range: 1 - 20
# ==========================

train_and_predict(

    "../data/BT2024081_train_var2(in).csv",

    "../data/BT2024081_test_var2(in).csv",

    "../predictions/BT2024081_pred_var2.csv",

    "../predictions/model_results_var2.csv",

    "../predictions/degree_vs_mse_var2.png",

    13

)