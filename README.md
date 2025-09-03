# Example of data projections in neural networks
## Purpose:  
Explore how neural nets perform transformations/representations of data to perform binary classification tasks. 

## Methods:  
1. Generate a synthetic data set: a 2-dimensional set of features (X<sub>1</sub> and X<sub>2</sub>) and a target variable, y, which is binary (class 0 or 1).
2. Force a relationship between X's and y; otherwise, it will be difficult to make predictions using just two features.
3. Build network. I chose to create this from scratch using resources from simple online tutorials...

**Number of layers:** 2 hidden, 1 output  
**Layer structure:** 3 neurons -> 3 neurons -> 1 neuron  
**Inputs:** 2   
**Full network architecture:** 2 inputs -> 3-dimensional representation -> 3-dimensional representation -> 1-dimensional probability (for classification task)  
**Loss:** Binary Cross-Entropy
**Optimization method:** Full-batch gradient descent

-----------
# Figures!  

**Figure 1:** Scatter plots of X<sub>1</sub> and X<sub>2</sub>, one labeled and one unlabeled, demonstrating a uniform distribution of points for both features and an inability to perfectly divide the classes using the two features as they are currently. 
| Scatter of features, unlabeled                                          | Scatter of features, labeled                                    |
|-------------------------------------------------------------------------|-----------------------------------------------------------------|
| <img src="figs/Figure_1.png" alt="2-d scatter of features" width="600"> | <img src="figs/Figure_5.png" alt="labeled scatter" width="600"> |

-----------

**Figure 2:** 3-D representation of the original inputs (X<sub>1</sub> and X<sub>2</sub>) after the first network layer.
<img src="figs/Figure_9.png" alt="3-d scatter of transformed values" width="600">  

-----------


**Figure 3:** 3-D representation of the original inputs (X<sub>1</sub> and X<sub>2</sub>) after the second network layer.
<img src="figs/Figure_10.png" alt="3-d scatter of transformed values" width="600">  

-----------

**Figure 4:** 1-D representation of the original inputs (X<sub>1</sub> and X<sub>2</sub>) after the third network layer. A grey dashed line is included as an example class separation line at 0.50.  
<img src="figs/Figure_11.png" alt="1-d line of transformed values" width="600">  

-----------

## Big picture take-away:  
- Neural networks create higher-level representations of the original inputs
- These representations help identify patterns in data that may be particularly difficult to find in high-dimensional data
- Transformation of data to these new representations is the task of weights (i.e., weighted sums at each neuron) plus the choice of activation function (i.g., sigmoid)
- A "good" representation of the data is one where (1) weights reflect a global minima in gradient descent, or at least a local minima that generalizes well, and (2) resulting representations improve overall prediction accuracy. 









