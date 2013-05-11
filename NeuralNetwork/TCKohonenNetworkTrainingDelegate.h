//
//  TCKohonenNetworkTrainingDelegate.h
//  NeuralNetwork
//
//  Created by theo on 5/6/13.
//  Copyright (c) 2013 Theodore Calmes. All rights reserved.
//

#import <Foundation/Foundation.h>

@class TCKohonenNetwork;

@protocol TCKohonenNetworkTrainingDelegate <NSObject>

@required

/** Asks the delegate the number of training examples you will be loading.

 @param network An object representing the neural network request this information.
 @return The number of examples to load.
 @see trainingOutputExamplesForNeuralNetwork:
 @see trainingInputExamplesForNeuralNetwork:
 */
- (NSInteger)numberOfTrainingExamplesForNeuralNetwork:(TCKohonenNetwork *)network;

/** Asks the delegate for the set of features for all the training examples.

 These values are represented as by a matrix of floats with m rows and n columns. m is the number of training examples, and n is the number of features.

 @param network An object representing the neural network request this information.
 @return Float matrix representation of the training examples' features.
 @see numberOfTrainingExamplesForNeuralNetwork:
 @see trainingInputExamplesForNeuralNetwork:
 */
- (float **)trainingInputExamplesForNeuralNetwork:(TCKohonenNetwork *)network;

@optional

/** Asks the delegate for the maximum number of iterations (epoch) gradient descent should perform.
 @param network An object representing the neural network request this information.
 @return An integer representing the stopping point for training the network. Default is 500.
 */
- (NSInteger)maxIterationsForNeuralNetwork:(TCKohonenNetwork *)network;

/** Callback for when the training examples are loaded.
 @param network An object representing the neural network request this information.
 */
- (void)neuralNetworkDidFinishLoadingTrainingExamples:(TCKohonenNetwork *)network;

/** Callback for when the training is completed.
 @param network An object representing the neural network request this information.
 */
- (void)neuralNetworkdidFinishTraining:(TCKohonenNetwork *)network;

/** Callback for when an individual step of gradient descent is completed.
 @param network An object representing the neural network request this information.
 @param epoch the total number of steps completed up till this point.
 @param costValue the networks current cost value.
 */
- (void)neuralNetwork:(TCKohonenNetwork *)network didCompleteTrainingEpoch:(NSInteger)epoch winningIndex:(float *)index;
@end
