//
//  TCNeuralNetwork.h
//
//  Created by Theodore Calmes on 11/18/12.
//  Copyright (c) 2012 Theodore Calmes. All rights reserved.
//

#import <Foundation/Foundation.h>

@class TCTheta;
@class TCFeedForwardNetwork;
@class TCFeedForwardNeuralNetworkTrainingParameters;

@protocol TCFeedForwardNetworkDelegate;
@protocol TCFeedForwardNetworkTrainingDelegate;

@interface TCFeedForwardNetwork : NSObject

/** Conform to this delegate if you want to initialize your weights from a pre-trained source.
 */
@property (weak, nonatomic) id<TCFeedForwardNetworkDelegate> delegate;

/** Conform to this delegate if you want to train a neural network.
 */
@property (weak, nonatomic) id<TCFeedForwardNetworkTrainingDelegate> trainingDelegate;

@property (strong, nonatomic) TCFeedForwardNeuralNetworkTrainingParameters *trainer;

/** This property holds the weights which define how your network has interpreted the set of training examples.
 */
@property (strong, readonly, nonatomic) TCTheta *weights;

/** Initializer which takes in an array of NSNumber integers. Eeach number represents a layer in the network, you need to provide atleast two numbers.
 @param neuronLayers This array represents the layers in your neural network. @[@100, @25, @10] would represent a network with 100 input units, 10 output units and 25 hidden layer units.
 @return A TCFeedForwardNetwork object.
 */
- (id)initWithLayers:(NSArray *)neuronLayers;

/** This method runs the input array through forward propagation to retrive a final output value.
 @param input Is an array of feature variables.
 @return The calculated output.
 */
- (float)classifyInput:(float *)input;

/** Call this method to load the data for your delegate. Depending on how you data is stored this could be a costly operation and should not be run on the main thread.
 */
- (void)loadDelegateData;

/** Runs gradient descent until either the cost difference reaches the threshold or the max number of iterations is reached.
 */
- (void)trainNetwork;

@end
