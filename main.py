"""
main.py
-------
ONE command runs the whole project end-to-end:

    python main.py

Pipeline:
  1. Generate synthetic HR data (virtual company simulation)
  2. Clean the data
  3. Engineer new features
  4. Exploratory data analysis (charts)
  5. Train & compare 4 ML models
  6. Evaluate the best model on unseen test data
  7. Generate HR insights + dashboard
  8. Predict performance for new employees
"""
import time

from src import config, data_generator, eda, evaluate, insights, predict, preprocessing, train


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main() -> None:
    start = time.time()
    config.make_dirs()

    banner("STEP 1/8  Generating synthetic employee data")
    data_generator.main()

    banner("STEP 2-3/8  Cleaning data & engineering features")
    df = preprocessing.main()

    banner("STEP 4/8  Exploratory data analysis")
    eda.main(df)

    banner("STEP 5/8  Training & comparing models")
    pipe, results, (X_train, X_test, y_train, y_test) = train.main(df)

    banner("STEP 6/8  Evaluating best model on test set")
    model_name = results.loc[0, "model"]
    metrics = evaluate.evaluate(pipe, X_test, y_test, model_name)
    evaluate.plot_model_comparison(results)
    fi = evaluate.plot_feature_importance(pipe, X_test, y_test)

    banner("STEP 7/8  Generating HR insights & dashboard")
    bundle = predict.load_model()
    insights.build_insights(df, bundle, fi, metrics)

    banner("STEP 8/8  Predicting performance for new employees")
    predict.main()

    banner(f"DONE in {time.time() - start:.1f}s  |  Best model: {model_name}  |  "
           f"Test accuracy: {metrics['test_accuracy']:.1%}")
    print("  Charts  -> images/")
    print("  Reports -> outputs/")
    print("  Model   -> models/best_model.joblib")
    print("  Dashboard (optional): streamlit run app.py\n")


if __name__ == "__main__":
    main()
