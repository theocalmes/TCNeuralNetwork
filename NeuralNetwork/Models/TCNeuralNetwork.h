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

@protocol TCNeuralNetworkDelegate;
@protocol TCNeuralNetworkTrainingDelegate;

@interface TCNeuralNetwork : NSObject

@property (weak, nonatomic) id<TCNeuralNetworkDelegate> delegate;
@property (weak, nonatomic) id<TCNeuralNetworkTrainingDelegate> trainingDelegate;
@property (strong, readonly, nonatomic) TCTheta *weights;

- (id)initWithLayers:(NSArray *)neuronLayers;
- (NSInteger)classifyInput:(float *)input;

- (void)loadDelegateData;
- (void)trainNetwork;

@end
