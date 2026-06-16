# Predicting the Remaining Useful Life (RUL) of Jet Turbofan Engines Using Depth-Continuous Neural ODEs

## Introduction

The aim of this project is to determine the efficacy of using neural ODEs (NODEs) to predict the Remaining Useful Life (RUL) of jet turbofan engines using the [CMAPSS Jet Engine Simulated Dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data). This is an important field of research as the consequences of unexpected jet engine failure are devastating such that it is important for engineers to know remaining service life estimates in order to enact proper preventative measures. Although there currently exist deep learning approaches–primarily LSTM or CNN based models–to predict RUL, they are founded upon a discrete-depth architecture, even though engine sensor readings are inherently time-continuous. Forcing a fixed time scale is unrealistic as oftentimes, real dynamic systems operate at different time scales due to the fact that for different data intervals, the data dynamics vary in frequency. This can lead to the loss of key temporal features of the data and introduce instability during back propagation, including the generation of vanishing or exploding gradients. Thus, the motivation behind the use of the NODEs stems from the fact that it is a depth-continuous neural network that replaces the discrete hidden layers with a vector field governed by an ordinary differential equation. The state of each "hidden layer" is then computed by integrating the ODE from $t_0$ to $t_1$ using numerical methods. As such, we hypothesize that state transformations in sensor data can be modeled by an NODE, where $$\frac{d \mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t, \theta)$$ and a neural network models the function $$f(\mathbf{h}(t), t, \theta)$$. The exploration of the CNN-NODE model, which prepends a CNN layer to the NODE, was driven by the motivation that the CNN will be able to pre-extract key patterns in the data that then lead to better RUL predictions once fed through the NODE. The CNN extracts key elements by convolving the weights of the kernel or convolution filter with the input data. By varying the weights of the kernel, the repeated convolution will be able to isolate key patterns within the sensor readings.

To evaluate the performance of our proposed model, two regression metrics commonly used in RUL prediction papers were used as the criteria to benchmark against the current state-of-the-art model: the root mean square error (RMSE), and the mean absolute percentage error (MAPE). The model architecture of the CNN-NODE and NODE can be found in [cnn_node.py](src/models/cnn_node.py) and [node.py](src/models/node.py), respectively, and the rest of the codebase is organized into files that govern specific functionalities.

### Outputting Graphs and Final Results

To generate graphical comparisons of the predicted and actual RULs of CNN-NODE and NODE, run [main.py](src/main.py) (note that the file must be run from the root directory). 8 graphs will be generated:

- The first 4 graphs show the CNN-NODE graphs in the order of FD001, FD002, FD003, FD004.
- The 4 graphs afterwards show the NODE graphs in the order of FD001, FD002, FD003, FD004.

The models' weights are stored in `.model` files under the [models](models/) directory. The [main.py](src/main.py) file loads the pre-trained model weigths into the specific type of model (CNN_NODE or NODE), evaluates RMSE and MAPE, and plots a graph comparing the predicted and ground truth RUL values.

### CMAPSS Jet Engine Simulated Data

The CMAPSS dataset is to be stored under the [CMAPSS](CMAPSS/) directory. Each FD00# file corresponds to a subset of data collected from simulated jet turbofan engine operations.

Files of the type "CMAPSS/RUL_FD00#.txt" consist of the ground truth measured RULs to validate the final trained models during testing.

Files of the type "CMAPSS/train_FD00#.txt" and "CMAPSS/test_FD00#.txt" hold sensor measurements for the 21 sensors at each time stamp and for the testing data specifically, each engine is run to failure (i.e. at the last time stamp for which data for a particular engine is recorded, the RUL is 0):  
Data is of size [total number of time cycles across all engines, 26]

- Column 1: unit number
- Column 2: time measured in number of cycles
- Columns 3 to 5: operational settings
- Column 6 to end: sensor measurements from the 21 sensors
- Columns 3-26 were used for training the models.

### Setting Up the Virtual Environment and Installing the Required Dependencies

1. In the command terminal, create a virtual environment through the following command: `python3 -m venv .venv`
2. Activate the virtual environment: `.venv\Scripts\activate` (for Windows) `source .venv/bin/activate` (for macOS/Linux)
3. Install the required libraries from requirements.txt: `pip install -r requirements.txt`
4. Select the Python interpreter to be the one inside .venv: `Ctrl + Shift + P`, choose `Select Interpreter` and choose the recommended option `.\.venv\Scripts\python.exe`

## Summary of Codebase Functionalities

### Data Processing

The data processing of the CMAPSS dataset is implemented in `src/utils/data_processing.py` and the methods are described below. It consists of three main components: Z-score standardization, RUL clipping, and sliding time window processing.

1. Z-Score Standardization
   Applying a Z-score standardization initially on the data enables meaningful comparisons between data values and reduces the disproportionate skew effects of large data values. Furthermore, by setting the mean to 0 and the standard deviation to 1, the data becomes evenly distributed around the origin with approximately 67% of the data within the range of `[-1, 1]`. It is important to normalize the input training data as neural networks work more efficiently with centered and scaled data. This is because during gradient descent, the gradient is calculated based on the magnitude of the data values, which means that the neural network weights are disproportionately altered and there are instances of vanishing and exploding gradients. This also makes it difficult to optimize hyperparameters such as the learning rate.
2. RUL Clipping
   All RUL values were clipped to a maximum of 100 cycles, as it is a sufficiently big number to adequately convey the idea that the current remaining lifespan of the engine is significantly long. Furthermore, this specific value was shown to work in the paper.
3. Sliding Time Window Processing
   Implementing a sliding window for the data set allows a better extraction of local temporal features and gives the model more data to make more robust predictions. A window size of `$k=40$` was chosen for benchmarking purposes as the paper also used this value.

### Neural ODE (NODE)

The code that defines the architecture of the NODE can be found in `src/models/node.py`.
Inheriting from the `nn.Module` base class, the `NeuralODE` class creates the architectural outline of a neural network by defining the derivative function $\frac{d \mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t, \theta)$ that follows a general discrete-depth feed-foreward architecture to generate the derivative for each state $h(t)$ over a continuous time interval. The `ODE` class performs numerical integration by calling the `NeuralODE` class to obtain the derivatives and evolving the current state to get the predicted values. The numerical solver used was the Dormand-Prince-5 (DOPRI-5, also known as RK45) method, which is widely used for numerically integrating ODEs where high precision is needed and adaptive time steps are used.

### CNN-NODE

The code defining the architecture of the CNN-NODE can be found by instancing the `CNN_ODE` class in `src/models/cnn_node.py` which also inherits from the `nn.Module` parent class. Due to the high RMSE values obtained from training the FD002 and FD004 datasets using the NODE architecture, a CNN layer was implemented to extract important features from engine data, before feeding it into the NODE. This would reduce the complexity of the data passed into the NODE, which would ultimately lead to more accurate RUL predictions.
The final optimized `CNN_ODE` class consists of a CNN layer of 36 kernels, each 3 integers long, that slides along the input data sequence one at a time such that no data value is skipped over. The final result from the CNN is then transformed into a vector in the input dimension of the NODE by the encoder, and is then passed to the NODE as the initial state. The output of the NODE is then obtained as previous mentioned.

### Model Utilities

Functions for initializing, training, and evaluating the different models can be found in `src/models/utils/model_utils.py`.

1. `def initialize_model` initializes the general architecture of the specific model (NODE or CNN-NODE) without the specification of any weights.

2. `def train_model` trains the model by implementing the Adam Algorithm to optimize and specify the weights of the model in order to minimize the loss function. Iterating through the epochs one at a time, and traversing through each epoch in batches, the function calls the current model under training to obtain its predicted values. The loss is then computed by comparing the predicted with the ground truth RUL values. The line `loss.backward()` then performs the back propagation step that determines how much each weight contributes to the total loss and calculates the gradient of the loss function with respect to each weight to determine the magnitude of weight adjustment.

3. `def train_model_repeated` trains multiple of the same type of models on incremental seeds.

4. `def evaluate_model` evaluates the model by comparing the model's predicted RUL with the ground truth RUL and graphs both in addition to calculating the RMSE and MAPE values. Note, that the RUL values are sorted in increasing order for ease of visual validation of the accuracy of the predicted RUL values and there is no relationship to be inferred from the resulting curves.

5. `def evaluate_model_repeated` evaluates multiple models of the same type.

### Tuning Hyperparameters

In `src/models/utils/tune_models.py`, models are trained based on the first 90% of the training data for each engine and are then evaluated on the rest of the 10% of the training data. It contains functions `learning_rate_sweep`, `hidden_dimensions_sweep`, and `dropout_rate_sweep` that sweep through different values of those respective hyperparameters to find the most optimized values that result in the lowest RMSE values. For example, in choosing the best learning rate, the `lr` entry of the settings/hyperparameter dictionary for the model is swapped with each of the five candidate learning rate values in the array in a for loop: `candidate_lrs = [0.1, 0.01, 0.005, 0.001, 0.0003]`

For the hidden dimensions, if the specific model being trained is CNN_NODE, the array is `[2, 3, 4]`; otherwise, if it is NODE, the array becomes `[32, 64, 128]`.

Furthermore, it is important to note that the seed is held constant throughout the process of training the models with different values of hyperparameters to make the results reproducible. Only then can lower calculated RMSE and MAPE values be attributed to changes in hyperparameter values and not due to randomly initialized values that happen to lead to favourable results.

### Constants

The `src/utils/constants.py` file contains all the constants used throughout all the files. Examples include two dictionaries `OPTIMIZED_CNN_NODE_SETTINGS` and `OPTIMIZED_NODE_SETTINGS` which correspond to the optimized hyperparameters of the CNN_NODE models and NODE models, respectively. They each describe the values of the hyperparameters to be used for their respective model. `OPTIMIZED_CNN_NODE_SETTINGS` also specifies the kernel size.
