# Cell Growth Rate Prediction Models
This project produces analyses of various degrees of nearest neighbor and polynomial regression models in determining cell growth rates.

## Highlights
 - Built a program that generates polynomial and kNN model comparisons.
 - Implemented train and test partitioning to ensure consistent evaluation.
 - Learned to analyze model data and distinguish improvements and data or model errors.
 - 

## Environment
- Python 3.13
- matplotlib 3.11.2
- numpy 2.5.3
- pandas 3.0.6
- scikit-learn 1.9.1

## Code Design 
### Imports
First the code imports the relevant packages, classes, and functions. 
Then it scans the dataset from the datasets folder. 
It then takes the dataset and splits it into target (the first column, the specific growth rate) which becomes y and features (T, pH, glucose, lactate) which becomes x. 
This stage is necessary because it sets up the functionality for the rest of the code and cuts the dataset into parts that can be read by the model training.

### Feature Subets


### Table Generation
At the end of the code is the table generation, which takes the data generated from the model training and testing and outputs it into a legible format, allowing me (and any interested parties) to analyze the results.
This stage is necessary because otherwise there would be no way for an interested party to see the results and compare models.


## Analysis
Due to the equation for distance weighting, the test data for the kNN with distance weighting is meaningless as the output is infinite when divided by 0 like in the training.
### Most Accurate Model
The most accurate model for predicting cell growth is the distance-weighted kNN with k=10. Reasonable alternatives are other kNN with distance weighting as they all perform above the mean, 
kNN with k=10 and uniform weighting, as it also performs highly in R2 and RSME for the test values,
some polynomial models also have high R2, however many higher degree polynomials are severely overfit, with R2 up to -11.
The reason the chosen model is better is 

### Conditions For Cell Growth
Increased cell growth can be accomplished by increasing the concentration of Glucose. 
The second degree polynomial (the one with the least negative R2) for exclusively glucose is 0.2654 +0.0274*[Glucose] +0.0074*[Glucose]^2.
Which implies a positive increase in specific growth rate in the presence of Glucose, which is to be expected as glucose is a carbon source for the cells to grow on.

Inversely, low cell growth can be accomplished by .


### Model Applicability
This model should be applicable for cells which function under similar conditions to those in the dataset used: temperatures from 30-42 degrees celcius, pH from 6-7.8, glucose concentrations from 0-39.9 mM, and Lactate concentrations from 0-40 mM. 
Many of the polynomial models predict behaviour which is impossible or unlikely outside of this range, for example the higher degree polynomials for glucose predict growth rates in the thousands just outside the data range.
The kNN may function slightly better at the edges of the existing data but is also expected to quickly lose accuracy once outside of the range of its data points.