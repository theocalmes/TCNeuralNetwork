//
//  TCNeuralNetwork.h
//  NeuralNetworks_TEST
//
//  Created by Theodore Calmes on 11/18/12.
//  Copyright (c) 2012 Theodore Calmes. All rights reserved.
//

#import <Foundation/Foundation.h>

@class TCTheta;
@class TCNeuralNetwork;

@protocol TCNeuralNetworkDelegate <NSObject>
@optional
- (float)regularizationParameterForNeuralNetwork:(TCNeuralNetwork *)network;
- (TCTheta *)weightsForNeuralNetwork:(TCNeuralNetwork *)network;

- (void)weightsDidLoadForNeuralNetwork:(TCNeuralNetwork *)network;

@end

@protocol TCNeuralNetworkTrainingDelegate <NSObject>
@required
- (NSInteger)numberOfTrainingExamplesForNeuralNetwork:(TCNeuralNetwork *)network;
- (NSInteger *)trainingOutputExamplesForNeuralNetwork:(TCNeuralNetwork *)network;
- (float **)trainingInputExamplesForNeuralNetwork:(TCNeuralNetwork *)network;

@optional
- (NSInteger)maxIterationsForNeuralNetwork:(TCNeuralNetwork *)network;
- (float)stopEpsilonForNeuralNetwork:(TCNeuralNetwork *)network;
- (float)learningParameterForNeuralNetwork:(TCNeuralNetwork *)network;

- (void)neuralNetworkDidFinishTraining:(TCNeuralNetwork *)network;
- (void)neuralNetworkDidFinishLoadingTrainingExamples:(TCNeuralNetwork *)network;
- (void)neuralNetwork:(TCNeuralNetwork *)network didCompleteTrainingEpoch:(NSInteger)epoch withCost:(float)costValue;

@end

@interface TCNeuralNetwork : NSObject

@property (weak, nonatomic) id<TCNeuralNetworkDelegate> delegate;
@property (weak, nonatomic) id<TCNeuralNetworkTrainingDelegate> trainingDelegate;
@property (strong, readonly, nonatomic) TCTheta *weights;

- (id)initWithLayers:(NSArray *)neuronLayers;
- (NSInteger)classifyInput:(float *)input;

- (void)loadDelegateData;
- (void)trainNetwork;

@end
