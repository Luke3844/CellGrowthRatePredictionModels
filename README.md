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
This stage is necessary because otherwise there would be no way for an interested party to see the results and compare models.


## Analysis
Due to the equation for distance weighting, the test data for the kNN with distance weighting is meaningless.
### Most Accurate Model
The most accurate model for predicting cell growth is the distance-weighted kNN with k=10. Reasonable alternatives are other kNN with distance weighting as they all perform above the mean, 
kNN with k=10 and uniform weighting, as it also performs highly in R2 and RSME for the test values.
The reason  is better is 

### Conditions For Cell Growth
Increased cell growth can be accomplished by increasing the concentration of Glucose. 
The second degree polynomial (the one with the least negative R2) for exclusively glucose is 0.2654 +0.0274*[Glucose] +0.0074*[Glucose]^2.
Which implies a positive increase in specific growth rate in the presence of Glucose, which is to be expected as glucose is a carbon source for the cells to grow on.

Inversely, low cell growth can be accomplished by .


### Model Applicability
This model should be applicable for cells which function under similar conditions to those in the dataset used: temperatures from 30-42 degrees celcius, pH from 6-7.8, glucose concentrations from 0-39.9 mM, and Lactate concentrations from 0-40 mM. 
