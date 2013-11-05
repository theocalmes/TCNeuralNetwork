//
//  TCFeedForwardNeuralNetworkTrainer.h
//  TBGPSTracking
//
//  Created by Theodore Calmes on 10/30/13.
//  Copyright (c) 2013 Theodore Calmes. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface TCFeedForwardNeuralNetworkTrainingParameters : NSObject

/** The regularization parameter (lambda) to use when calculating the cost (J).
 
    Cost += (lambda/2*m)sum(weights[l][i][j]^2, j={0,s(l+1)}, i={0,sl}, l={0,L-1}) where L is the number of layers, s(l+1) is units in layer l+1, and sl is units in layer l. Default is 1.0.

    This value is vital to getting good performace from your network. Try changing this value if your network is having trouble learning.
 
    Default = 1.0;
 */
@property (assign, nonatomic) float regularizationParameter;

/** The value (alpha) which determines the scale for the adjustment of each iteration of gradient descent.
 
    weight[l][i][j] := weight[l][i][j] - alpha * ∇J(weights)
 
    Default = 1.0
 */
@property (assign, nonatomic) float learningRate;

/** The maximum number of epoches (iterations of gradient descent).
 
    Default is 500.
*/
@property (assign, nonatomic) NSUInteger maxNumberOfIterations;

/** The value where the difference (newJ - oldJ) in cost should signal the end of training.

    Default is 0.00003
 */
@property (assign, nonatomic) float stopEpsilon;

/** Initialize the Network trainer with an example training data.
 
 @param inputExamples Float matrix representation of the training examples' features.
 @param outputExamples Float array usually representing a tag for a class.
 @param count The total number of examples.
 @return An instance of a TCFeedForwardNeuralNetworkTrainer to use for training a TCFeedForwardNeuralNetwork.
 @see numberOfTrainingExamples
 @see trainingOutputExamples
 @see trainingInputExamples
 */
- (id)initWithTrainingInputExamples:(float **)inputExamples trainingOutputExamples:(float *)outputExamples count:(NSInteger)count;

/** The number of training examples you will be loading.

    Note: this value is set durning the initialization.

 @return The number of examples to load.
 @see initWithTrainingInputExamples:trainingOutputExamples:count:
 @see trainingOutputExamples
 @see trainingInputExamples
 */
- (NSInteger)numberOfTrainingExamples;

/** The set of features for all the training examples.

 These values are represented as by a matrix of floats with m rows and n columns. m is the number of training examples, and n is the number of features.
 
 Note: This value is set in the init method.

 @return Float matrix representation of the training examples' features.
 @see numberOfTrainingExamples
 @see trainingOutputExamples
 */
- (float **)trainingInputExamples;

/** Asks the delegate for the set of output variables for all the training examples.

 If you are making a classifier (output size > 1) these values are represented by an float array where each float is casted to an integer value and represents a separate class i.e you are trying to do optical character recognition, 0 would map to A, 1 to B and so on. If you have just a single output neuron you can have this return an array of floats as is.
 
 Note: This value is set in the init method.

 @return Float array of size numberOfTrainingExamples.
 @see numberOfTrainingExamples
 @see trainingInputExamples
 */
- (float *)trainingOutputExamples;

@end
