//
//  TCFeedForwardNeuralNetworkTrainer.m
//  TBGPSTracking
//
//  Created by Theodore Calmes on 10/30/13.
//  Copyright (c) 2013 Theodore Calmes. All rights reserved.
//

#import "TCFeedForwardNeuralNetworkTrainingParameters.h"

@implementation TCFeedForwardNeuralNetworkTrainingParameters
{
    NSInteger m;
    float *y;
    float **X;
}

- (id)initWithTrainingInputExamples:(float **)inputExamples trainingOutputExamples:(float *)outputExamples count:(NSInteger)count
{
    self = [super init];
    if (self) {
        m = count;
        y = outputExamples;
        X = inputExamples;

        _learningRate = 1.0;
        _regularizationParameter = 1.0;
        _stopEpsilon = 0.00003;
        _maxNumberOfIterations = 500;
    }
    return self;
}

- (NSInteger)numberOfTrainingExamples
{
    return m;
}

- (float *)trainingOutputExamples
{
    return y;
}

- (float **)trainingInputExamples
{
    return X;
}

@end
