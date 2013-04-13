//
//  TCNeuralNetworkTrainingDelegate.h
//  NeuralNetwork
//
//  Created by theo on 4/12/13.
//  Copyright (c) 2013 Theodore Calmes. All rights reserved.
//

#import <Foundation/Foundation.h>

@class TCTheta;
@class TCNeuralNetwork;

/** The TCNeuralNetworkTrainingDelegate protocol provides you with methods to load a set of training data for the training process. The protocol also lets you customize the gradient descent parameters and provides methods to get feedback on the networks training.
 
    Note: The data is loaded by calling loadDelegateData on the neural network.
 */
@protocol TCNeuralNetworkTrainingDelegate <NSObject>

@required

/** Asks the delegate the number of training examples you will be loading.

 @param network An object representing the neural network request this information.
 @return The number of examples to load.
 @see trainingOutputExamplesForNeuralNetwork:
 @see trainingInputExamplesForNeuralNetwork:
 */
- (NSInteger)numberOfTrainingExamplesForNeuralNetwork:(TCNeuralNetwork *)network;

/** Asks the delegate for the set of output variables for all the training examples.
 
 These values are represented by an integer array where each integer represents a separate class i.e you are trying to do optical character recognition, 0 would map to A, 1 to B and so on.

 @param network An object representing the neural network request this information.
 @return Integer array of size numberOfTrainingExamplesForNeuralNetwork:.
 @see numberOfTrainingExamplesForNeuralNetwork:
 @see trainingInputExamplesForNeuralNetwork:
 */
- (NSInteger *)trainingOutputExamplesForNeuralNetwork:(TCNeuralNetwork *)network;

/** Asks the delegate for the set of features for all the training examples.

 These values are represented as by a matrix of floats with m rows and n columns. m is the number of training examples, and n is the number of features.

 @param network An object representing the neural network request this information.
 @return Float matrix representation of the training examples' features.
 @see numberOfTrainingExamplesForNeuralNetwork:
 @see trainingInputExamplesForNeuralNetwork:
 */
- (float **)trainingInputExamplesForNeuralNetwork:(TCNeuralNetwork *)network;

@optional

/** Asks the delegate for the regularization parameter (lambda) to use when calculating the cost (J).

 This value is vital to getting good performace from your network. Try changing this value if your network is having trouble learning.

 @param network An object representing the neural network request this information.
 @return a float which will be used in the calculation of the cost. Cost += (lambda/2*m)sum(weights[l][i][j]^2, j={0,s(l+1)}, i={0,sl}, l={0,L-1}) where L is the number of layers, s(l+1) is units in layer l+1, and sl is units in layer l. Default is 1.0.
 */
- (float)regularizationParameterForNeuralNetwork:(TCNeuralNetwork *)network;

/** Asks the delegate for the maximum number of iterations (epoch) gradient descent should perform.
 @param network An object representing the neural network request this information.
 @return An integer representing the stopping point for training the network. Default is 500.
 */
- (NSInteger)maxIterationsForNeuralNetwork:(TCNeuralNetwork *)network;

/** Asks the delegate for the value where the difference (newJ - oldJ) in cost should signal the end of training.
 @param network An object representing the neural network request this information.
 @return A float value. Default is 0.00003
 */
- (float)stopEpsilonForNeuralNetwork:(TCNeuralNetwork *)network;

/** Asks the delegate for the value which determines the scale for the adjustment of each iteration of gradient descent.
 @param network An object representing the neural network request this information.
 @return A float value (alpha) used by gradient descent. weight[l][i][j] := weight[l][i][j] - alpha * ∇J(weights)
 */
- (float)learningParameterForNeuralNetwork:(TCNeuralNetwork *)network;

/** Callback for when the training is completed.
 @param network An object representing the neural network request this information.
 */
- (void)neuralNetworkDidFinishTraining:(TCNeuralNetwork *)network;

/** Callback for when the training examples are loaded.
 @param network An object representing the neural network request this information.
 */
- (void)neuralNetworkDidFinishLoadingTrainingExamples:(TCNeuralNetwork *)network;

/** Callback for when an individual step of gradient descent is completed.
 @param network An object representing the neural network request this information.
 @param epoch the total number of steps completed up till this point.
 @param costValue the networks current cost value.
 */
- (void)neuralNetwork:(TCNeuralNetwork *)network didCompleteTrainingEpoch:(NSInteger)epoch withCost:(float)costValue;

@end
