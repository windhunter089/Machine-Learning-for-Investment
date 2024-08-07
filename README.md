# Trading Strategy Evaluation Using Bagging Learners with Technical Analysis Indicators

## Overview
This project evaluates the effectiveness of a manual trading strategy versus a machine learning-based trading strategy. Both strategies use three technical indicators: EMA9 crossover EMA21, MACD, and Bollinger Bands. The machine learning strategy employs a bagging learner with decision trees.

A part of the project involve evaluate machine learning leaners like regression, decision tree, random forest and bagging.

Another part of the project involve indicators evaluation (EMA crossover, Bollinger Band Percentage, MACD, Stochastic Oscillator, Rate of Change)


## Project Structure
- **strategy_evaluation/**: Source code and detailed result on the evaluation process. Experiment 1 is comparison of manual strategy vs machine learning. Experiment 2 is sensitity test.
  - **testproject.py**: combination of all experiments for the trading evaluation. ```python testproject.py ```
- **indicator_evaluation/**: Source code for evaluation of trading indicators.
  - **manual_strategy.py**: Implementation of the manual trading strategy.
  - **strategy_learner.py**: Implementation of the machine learning trading strategy.
  - **evaluate.py**: Script to compare the performance of both strategies.
- **assess_learners/**: Source code for evaluation of different machine learning leaners.
  - **manual_strategy.py**: Implementation of the manual trading strategy.
  - **strategy_learner.py**: Implementation of the machine learning trading strategy.
  - **evaluate.py**: Script to compare the performance of both strategies.
- **data/**: Contains the historical stock price data used for training and testing the models.
- **grading/**: Grade scope intergration

## Indicators Overview
### EMA9 Crossover EMA21
The EMA9 crossover EMA21 is a popular technical analysis tool. A positive difference between EMA9 and EMA21 indicates an uptrend, while a negative difference indicates a downtrend.

### MACD
MACD consists of the MACD line and the signal line. The MACD histogram, which is the difference between these two lines, is used to identify trend changes, momentum, and potential buy or sell signals.

### Bollinger Bands Percentage (BBP)
Bollinger Bands measure stock price volatility. BBP indicates the position of the stock price relative to the bands, with values over 70 suggesting overbought conditions and values below 30 suggesting oversold conditions.

## Manual Strategy
The manual strategy follows the trend indicated by the EMA crossover and MACD histogram, avoiding trades when BBP indicates overbought or oversold conditions.

## Strategy Learner
The strategy learner uses a bagging method with decision trees to predict buy, sell, or hold signals based on the continuous values of the three indicators. The model is trained on in-sample data and tested on out-of-sample data.

## Performance Comparison
### In-Sample Performance
- **Manual Strategy**: Outperformed the benchmark with significant gains.
- **Strategy Learner**: Outperformed both the manual strategy and the benchmark, indicating the effectiveness of machine learning in trading.

### Out-of-Sample Performance
- **Manual Strategy**: Performance dropped but still outperformed the benchmark.
- **Strategy Learner**: Performance remained strong, significantly outperforming both the manual strategy and the benchmark.

## Conclusion
Using technical indicators in a machine learning model like the strategy learner provided outstanding results. However, more testing is needed to consider other factors such as entry price and commission fees.



### Author
Feedback and question, please forward to Ted Pham - trungpham89@gmail.com