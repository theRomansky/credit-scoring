# Credit Scoring Classification Project

This repository contains a data science project focused on building a two-class classification model for credit scoring. The primary goal is to predict the creditworthiness of individuals based on various features. The implemented pipeline includes data extraction, feature engineering, encoding, aggregation, and model training using the CatBoost classifier.

## Project Structure

The project follows a structured pipeline, as outlined below:

1. **Data Extraction:**
   - The `extraction` function reads data from Parquet files, excluding the 'rn' column.

2. **Feature Engineering:**
   - The `generate_new_features` function creates new features such as `total_delinquencies` and `total_undefined_days`.

3. **Data Encoding:**
   - The `encoding` function utilizes OneHotEncoder to transform categorical features.

4. **Data Aggregation:**
   - The `group_aggregate` function aggregates data based on the 'id' column using a specified aggregation function.

5. **Data Processing:**
   - The `process_data_chunk` function applies the entire pipeline to individual data chunks, progressively building the dataset.

6. **Model Training:**
   - The CatBoost classifier is used in a scikit-learn pipeline (`catboost_pipeline`) to train the final model.

7. **Model Serialization:**
   - The trained model is serialized using joblib and saved as `catboost_pipeline.pkl`.

## Dependencies

- pandas
- scikit-learn
- catboost

## Author

Roman Kovalenko
