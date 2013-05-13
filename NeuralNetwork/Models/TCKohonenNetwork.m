//
//  TCKohonenNetwork.m
//  NeuralNetwork
//
//  Created by theo on 5/5/13.
//  Copyright (c) 2013 Theodore Calmes. All rights reserved.
//

#import "NeuralNetwork.h"

void deallocNeuron(Neuron neuron)
{
    free(neuron.weights);
    free(neuron.index);
}

Neuron newNeuron(float *index, NSInteger indexLength, NSInteger weightsLength, TCRange range)
{
    Neuron new;
    new.index = index;
    new.indexLength = indexLength;
    new.weightsLength = weightsLength;
    new.weights = malloc(weightsLength * sizeof(float));

    for (NSInteger i = 0; i < weightsLength; i++) {
        new.weights[i] = randomRange(range);
    }

    return new;
}

Neuron copyNeuron(Neuron neuron)
{
    Neuron new;
    new.index = (float *)calloc(neuron.indexLength, sizeof(float));
    new.indexLength = neuron.indexLength;
    vDSP_vadd(new.index, 1, neuron.index, 1, new.index, 1, neuron.indexLength);

    new.weights = (float *)calloc(neuron.weightsLength, sizeof(float));
    new.weightsLength = neuron.weightsLength;
    vDSP_vadd(new.weights, 1, neuron.weights, 1, new.weights, 1, neuron.weightsLength);

    return new;
}

float lateralDistance(Neuron n1, Neuron n2)
{
    assert(n1.indexLength == n2.indexLength);

    float val = 0.0;
    NSInteger len = n1.indexLength;

    float *diff = malloc(len * sizeof(float));
    vDSP_vsub(n1.index, 1, n2.index, 1, diff, 1, len);
    vDSP_dotpr(diff, 1, diff, 1, &val, len);

    free(diff);

    return val;
}

float decay(float a0, float t0, float t)
{
    return a0 * expf(-t/t0);
}

@implementation TCKohonenNetwork
{
    Neuron *neurons;

    NSInteger N; // Number of neurons
    NSInteger M; // Number of features

    float nS0; // Initial neighbourhood size
    float nST; // Neighbourhood size time decay

    float lR0; // Initial learning rate
    float lRT; // Learning rate time decay

    float **samples;
    NSInteger numSamples;

    NSInteger maxIterations;
}

- (id)initWithInputLayerDimension:(NSInteger)dimension
{
    self = [super init];
    if (!self) return nil;

    M = dimension;

    nS0 = 0.5;
    nST = 0.5;
    lR0 = 0.5;
    lRT = 0.5;

    maxIterations = 500;

    return self;
}

#pragma mark - Topology Setup

- (void)setupNeuronsUsing2DGridTopologyWithWidth:(NSInteger)width height:(NSInteger)height randomRange:(TCRange)range
{
    N = width * height;
    neurons = malloc(N * sizeof(Neuron));
    NSInteger count = 0;

    for (NSInteger i = 0; i < height; i++) {
        for (NSInteger j = 0; j < width; j++) {

            float *index = malloc(2 * sizeof(float));
            index[0] = i;
            index[1] = j;

            neurons[count] = newNeuron(index, 2, M, range);
            count++;
        }
    }
}

- (void)setupNeuronsUsing3DGridTopologyWithWidth:(NSInteger)width height:(NSInteger)height depth:(NSInteger)depth randomRange:(TCRange)range
{
    N = width * height * depth;
    neurons = malloc(N * sizeof(Neuron));
    NSInteger count = 0;

    for (NSInteger i = 0; i < height; i++) {
        for (NSInteger j = 0; j < width; j++) {
            for (NSInteger k = 0; k < depth; k++) {

                float *index = malloc(3 * sizeof(float));
                index[0] = i;
                index[1] = j;
                index[2] = k;

                neurons[count] = newNeuron(index, 3, M, range);
                count++;
            }
        }
    }
}

- (void)setupNeuronsUsingCustomTopologyWithIndices:(float **)indices indexSize:(NSInteger)size numberOfNeurons:(NSInteger)neuronCount randomRange:(TCRange)range
{
    N = neuronCount;
    neurons = malloc(N * sizeof(Neuron));

    for (NSInteger i = 0; i < N; i++) {
        neurons[i] = newNeuron(indices[i], size, M, range);
    }
}

#pragma mark - Getters & setters

- (Neuron *)neurons
{
    return neurons;
}

- (void)setLearningRate:(float)klR0 learningDecay:(float)klRT neighbourhoodSize:(float)knS0 sizeDecay:(float)knST
{
    lR0 = klR0;
    lRT = klRT;
    nS0 = knS0;
    nST = knST;
}

#pragma mark - Setup From Delegate

- (void)setupNetwork
{
    //
}

- (void)loadTrainingSet
{
    samples = [self.trainingDelegate trainingInputExamplesForNeuralNetwork:self];
    numSamples = [self.trainingDelegate numberOfTrainingExamplesForNeuralNetwork:self];

    if ([self.trainingDelegate respondsToSelector:@selector(maxIterationsForNeuralNetwork:)])
        maxIterations = [self.trainingDelegate maxIterationsForNeuralNetwork:self];

    if ([self.trainingDelegate respondsToSelector:@selector(neuralNetworkDidFinishLoadingTrainingExamples:)])
        [self.trainingDelegate neuralNetworkDidFinishLoadingTrainingExamples:self];
}

#pragma mark - Training

- (void)dealloc
{
    for (NSInteger i = 0; i < N; i++) {
        deallocNeuron(neurons[i]);
    }
    free(neurons);
}

- (Neuron)winningNeuronForInput:(float *)input
{
    NSInteger index = 0;
    float min = FLT_MAX;

    for (NSInteger i = 0; i < N; i++) {

        Neuron neuron = neurons[i];
        float *diff = malloc(M * sizeof(float));
        vDSP_vsub(input, 1, neuron.weights, 1, diff, 1, M);
        float dotProduct = 0.0;
        vDSP_dotpr(diff, 1, diff, 1, &dotProduct, M);
        dotProduct = sqrtf(dotProduct);

        if (dotProduct < min) { min = dotProduct; index = i; };

        free(diff);
    }

    return neurons[index];
}

- (float *)topologicalNeighbourhoodForWinningNeuron:(Neuron)winner atTime:(float)t
{
    float *T = malloc(N * sizeof(float));
    float variance = 2*powf(decay(nS0, nST, t), 2);

    for (NSInteger i = 0; i < N; i++) {
        float distance = lateralDistance(winner, neurons[i]);

        T[i] = expf(-distance / variance);
    }

    return T;
}

- (void)updateWeightsUsingInput:(float *)x atTime:(float)t
{
    Neuron winner = [self winningNeuronForInput:x];
    float *T = [self topologicalNeighbourhoodForWinningNeuron:winner atTime:t];
    float L = decay(lR0, lRT, t);

    for (NSInteger i = 0; i < N; i++) {
        for (NSInteger j = 0; j < M; j++) {
            neurons[i].weights[j] = neurons[i].weights[j] + L * T[i] * (x[j] - neurons[i].weights[j]);
        }
    }

    free(T);
}

#pragma mark - Public Methods

- (void)loadDelegateData
{
    if (self.delegate)
        [self setupNetwork];

    if (self.trainingDelegate)
        [self loadTrainingSet];
}

- (void)trainNetworkForTimeRange:(TCRange)timeRange
{
    if (!self.trainingDelegate) return;

    BOOL feedback = [self.trainingDelegate respondsToSelector:@selector(neuralNetwork:didCompleteTrainingEpoch:winningIndex:)];

    NSInteger start = (NSInteger)timeRange.low;
    NSInteger end = (NSInteger)timeRange.high;
    
    for (NSInteger t = start; t < end; t++) {

        NSInteger randIndex = arc4random() % numSamples;
        Neuron winner = [self winningNeuronForInput:samples[randIndex]];

        [self updateWeightsUsingInput:samples[randIndex] atTime:t];

        if (feedback)
            [self.trainingDelegate neuralNetwork:self didCompleteTrainingEpoch:t winningIndex:winner.index];
    }
}

- (void)trainNetwork
{
    if (!self.trainingDelegate) return;

    [self trainNetworkForTimeRange:TCRangeMake(0, maxIterations)];
    
    if ([self.trainingDelegate respondsToSelector:@selector(neuralNetworkdidFinishTraining:)])
        [self.trainingDelegate neuralNetworkdidFinishTraining:self];
}





@end
