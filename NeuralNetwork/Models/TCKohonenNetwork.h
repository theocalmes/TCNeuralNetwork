//
//  TCKohonenNetwork.h
//  NeuralNetwork
//
//  Created by theo on 5/5/13.
//  Copyright (c) 2013 Theodore Calmes. All rights reserved.
//

#import <Foundation/Foundation.h>

@protocol TCKohonenNetworkDelegate;
@protocol TCKohonenNetworkTrainingDelegate;

typedef struct Neuron {
    float *index;
    NSInteger indexLength;
    float *weights;
    NSInteger weightsLength;
} Neuron;

@interface TCKohonenNetwork : NSObject
@property (weak, nonatomic) id<TCKohonenNetworkDelegate> delegate;
@property (weak, nonatomic) id<TCKohonenNetworkTrainingDelegate> trainingDelegate;

- (id)initWithInputLayerDimension:(NSInteger)dimension;
- (Neuron *)neurons;

- (void)setLearningRate:(float)klR0 learningDecay:(float)klRT neighbourhoodSize:(float)knS0 sizeDecay:(float)knST;

- (void)setupNeuronsUsing2DGridTopologyWithWidth:(NSInteger)width height:(NSInteger)height;
- (void)setupNeuronsUsing3DGridTopologyWithWidth:(NSInteger)width height:(NSInteger)height depth:(NSInteger)depth;
- (void)setupNeuronsUsingCustomTopologyWithIndices:(float **)indices indexSize:(NSInteger)size numberOfNeurons:(NSInteger)neuronCount;

- (void)loadDelegateData;
- (void)trainNetwork;

@end
