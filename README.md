# Cell Growth Rate Prediction Models
This project produces analyses of various degrees of nearest neighbor and polynomial regression models in determining cell growth rates.

## Highlights
 - Built a resuable pipeline that generates polynomial and kNN model comparisons.
 - Implemented train and test partitioning to ensure consistent evaluation.
 - Learned to analyze model data and distinguish improvements and data or model errors.

## Environment 
- Python 3.14.7
- Add rest later

## Code Design 
First the code imports the relevant packages and scans the dataset from the datasets folder. Then the parameters for the 

At the end is the table generation, which takes the data generated from the model training and testing and outputs it into a legible format, allowing me (and any interested parties) to analyze the results.

## Analysis

### Most Accurate Model
The most accurate model for predicting cell growth is . Reasonable alternatives are . The reason  is better is 

### Conditions For Cell Growth
High cell growth can be accomplished by .

Inversely, low cell growth can be accomplished by .


### Model Applicability
This model should be applicable for cells which function under similar conditions to those in the dataset used: temperatures from 30-42 degrees celcius, pH from 6-7.8, glucose concentrations from 0-39.9 mM, and Lactate concentrations from 0-40 mM. 
