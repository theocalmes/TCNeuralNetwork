//
//  TCNeuralNetworkTrainingDelegate.h
//  NeuralNetwork
//
//  Created by theo on 4/12/13.
//  Copyright (c) 2013 Theodore Calmes. All rights reserved.
//

#import <Foundation/Foundation.h>

@class TCTheta;
@class TCFeedForwardNetwork;

/** The TCFeedForwardNetworkTrainingDelegate protocol provides you with callbacks to observe the training of a neural network.
 */
@protocol TCFeedForwardNetworkTrainingDelegate <NSObject>

@optional

/** Callback for when the training is completed.
 @param network An object representing the neural network request this information.
 @param costValue The cost at the last training step.
 */
- (void)neuralNetworkDidFinishTraining:(TCFeedForwardNetwork *)network;

/** Callback for when an individual step of gradient descent is completed.
 @param network An object representing the neural network request this information.
 @param epoch the total number of steps completed up till this point.
 @param costValue the networks current cost value.
 */
- (void)neuralNetwork:(TCFeedForwardNetwork *)network didCompleteTrainingEpoch:(NSInteger)epoch withCost:(float)costValue;

@end
