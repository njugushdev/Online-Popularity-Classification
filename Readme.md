# Predicting Online News Popularity Using Machine Learning

## Overview
This project delivers a machine learning pipeline crafted to predict the popularity of online news articles, defined as achieving over 1400 shares, leveraging the Online News Popularity dataset from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/332/online+news+popularity). the project employs three distinct algorithms—Naïve Bayes, Support Vector Machine (SVM), and Random Forest—to classify articles based on content and metadata features. The pipeline evaluates performance through accuracy, F1-score, precision, recall, confusion matrices, and learning curves, utilizing 5-fold cross-validation and 3-fold nested cross-validation for robust hyperparameter tuning. Designed for researchers and practitioners interested in digital media analytics, this project showcases a systematic approach to understanding the drivers of online content virality.

## Technical Details
The project is implemented in a Jupyter notebook (`Online_Network_Polarity_Notes.ipynb`) executed locally using Python, processing a dataset of 39,644 articles with 60 features, including word count, sentiment polarity, and publication channel. Preprocessing Hyperparameter tuning employs GridSearchCV with 3-fold nested cross-validation, optimizing parameters like `var_smoothing` for Naïve Bayes, `C` and kernel type for SVM, and `n_estimators` and `max_depth` for Random Forest. The project organizes code in a `notebooks/` folder, stores the dataset in a `data/` folder, and saves output visualizations (e.g., histograms, confusion matrices) in a `results/` folder, ensuring a modular and reproducible structure.

## How to Run the Project
To explore this project locally, clone the repository from Git and ensure Python 3.8+ is installed. Navigate to the project root and install dependencies from `requirements.txt` using the command `pip install -r requirements.txt`, which includes pandas, scikit-learn, seaborn, matplotlib, and jupyter for data processing, modeling, and visualization. The dataset (`OnlineNewsPopularity.csv`) is downloaded automatically in Cell 3 of `Online_Network_Polarity_Notes.ipynb`, or it can be placed in the `data/` folder if preferred. Open the notebook in Jupyter Notebook by running `jupyter notebook` and execute cells sequentially to preprocess data, train models, and generate results. Key cells include Cell 6 for hyperparameter tuning, which may take 10–20 minutes depending on hardware, Cell 7 for computing test metrics and confusion matrices, and Cell 8 for plotting learning curves, requiring about 5–10 minutes. For optimal performance, use a machine with at least 8GB RAM and a multi-core CPU; a GPU is not required but may accelerate SVM tuning if configured with a compatible backend. Outputs, such as histograms and confusion matrices, are saved to the `results/` folder for easy access.

## Project Significance
This project exemplifies a robust approach to machine learning for digital media analytics, addressing the challenge of predicting online news popularity with a carefully designed pipeline. By integrating feature engineering, such as sentiment ratios and channel indicators, with rigorous evaluation through nested cross-validation and learning curves, the project ensures reliable and interpretable results. The comparison of high-bias (Naïve Bayes), medium-bias (SVM), and low-bias (Random Forest) algorithms highlights trade-offs in model complexity, offering insights into their suitability for content virality prediction. The modular structure, with clear separation of notebooks, data, and results, facilitates reproducibility and extension, making this project a valuable resource for researchers and practitioners exploring machine learning applications in media and social engagement analysis.


## Disclaimer

**Copyrights Reserved**:  
@njugushdev  

This project, including its code and resources, is intended solely for educational purposes and should not be used for any commercial purposes without proper authorization.
