//
//  TCNeuralNetwork.m
//  NeuralNetworks_TEST
//
//  Created by Theodore Calmes on 11/18/12.
//  Copyright (c) 2012 Theodore Calmes. All rights reserved.
//

#import "NeuralNetwork.h"

float sigmoid(float z)
{
    return (1.0/(1.0+expf(-z)));
}

float gradSigmoid(float z)
{
    return sigmoid(z)*(1.0-sigmoid(z));
}

float gradient_activity(float a)
{
    return a*(1.0-a);
}

float *forwardPropagate(float *aj, TCTheta *theta, NSInteger l)
{
    TCDimension WlDim = [theta dimensionForLayer:l];
    int M = WlDim.rows;
    int N = WlDim.cols;

    float *a __attribute__ ((aligned)) = (float *)calloc(N, sizeof(float));
    a[0] = 1.0;
    for (int i=1; i<N; ++i) a[i] = aj[i-1];

    float *next __attribute__ ((aligned)) = (float *)calloc(M, sizeof(float));
    vDSP_mmul(theta.weights[l], 1, a, 1, next, 1, M, 1, N);

    for (int i=0; i<M; ++i)
        next[i] = sigmoid(next[i]);

    free(a);

    return next;
}

float *computeDelta(float *a, float *nextDelta, TCTheta *theta, NSInteger l)
{
    TCDimension WlDim = [theta dimensionForLayer:l];
    int M = WlDim.rows;
    int N = WlDim.cols;

    float *transposed __attribute__ ((aligned))  = (float*)calloc(N*M, sizeof(float));
    vDSP_mtrans(theta.weights[l], 1, transposed, 1, N, M);

    float *product __attribute__ ((aligned)) = (float*)calloc(N, sizeof(float));
    vDSP_mmul(transposed, 1, nextDelta, 1, product, 1, N, 1, M);

    float *delta __attribute__ ((aligned)) = (float*)calloc(N-1, sizeof(float));

    for (int i=1; i<N; ++i) delta[i-1] = product[i] * gradient_activity(a[i-1]);

    free(transposed);
    free(product);

    return delta;
}

void updateThetaGradient(TCTheta **theta, NSInteger l, float *al, float *deltaNext)
{
    TCDimension WlDim = [*theta dimensionForLayer:l];
    int M = WlDim.rows;
    int N = WlDim.cols;

    float *a __attribute__ ((aligned)) = (float *)calloc(N, sizeof(float));
    a[0] = 1.0;
    for (int i=1; i<N; i++) a[i] = al[i-1];
    
    float *transposed __attribute__ ((aligned)) = (float *)calloc(N, sizeof(float));
    vDSP_mtrans(a, 1, transposed, 1, 1, N);

    float *product __attribute__ ((aligned)) = (float *)calloc(M*N, sizeof(float));
    vDSP_mmul(deltaNext, 1, transposed, 1, product, 1, M, N, 1);

    free(transposed);
    free(a);
    
    [*theta addMatrix:product toLayer:l];
}

NSInteger *labelArray(NSInteger label, NSInteger total)
{
    NSInteger *array = (NSInteger *)calloc(total, sizeof(NSInteger));
    array[label] = 1;
    return array;
}

@interface TCNeuralNetwork ()
@property (strong, readwrite, nonatomic) TCTheta *weights;
@end

@implementation TCNeuralNetwork
{
    float **X;
    NSInteger *y;

    NSInteger m;
    NSInteger L;
    NSInteger numLabels;
    NSInteger maxIterations;

    float stopEpsilon;
    float learningParam;
    float lambda;
}

- (id)initWithLayers:(NSArray *)neuronLayers
{
    self = [super self];
    if (self) {
        _weights = [[TCTheta alloc] initWithLayers:neuronLayers];
        [_weights randomizeValuesWithEpsilon:0.12];

        L = neuronLayers.count;
        numLabels = [neuronLayers.lastObject integerValue];

        lambda = 1.0;
        maxIterations = 500;
        stopEpsilon = 0.00003;
        learningParam = 1.0;
    }
    return self;
}

#pragma mark - Setup From Delegate

- (void)setupNetwork
{
    if ([self.delegate respondsToSelector:@selector(weightsForNeuralNetwork:)]) {
        TCTheta *temp = [self.delegate weightsForNeuralNetwork:self];
        if (temp) {
            self.weights = temp;

            if ([self.delegate respondsToSelector:@selector(weightsDidLoadForNeuralNetwork:)])
                [self.delegate weightsDidLoadForNeuralNetwork:self];
        }
    }
}

- (void)loadTrainingSet
{
    X = [self.trainingDelegate trainingInputExamplesForNeuralNetwork:self];
    y = [self.trainingDelegate trainingOutputExamplesForNeuralNetwork:self];

    m = [self.trainingDelegate numberOfTrainingExamplesForNeuralNetwork:self];

    if ([self.trainingDelegate respondsToSelector:@selector(regularizationParameterForNeuralNetwork:)])
        lambda = [self.trainingDelegate regularizationParameterForNeuralNetwork:self];

    if ([self.trainingDelegate respondsToSelector:@selector(maxIterationsForNeuralNetwork:)])
        maxIterations = [self.trainingDelegate maxIterationsForNeuralNetwork:self];

    if ([self.trainingDelegate respondsToSelector:@selector(stopEpsilonForNeuralNetwork:)])
        stopEpsilon = [self.trainingDelegate stopEpsilonForNeuralNetwork:self];

    if ([self.trainingDelegate respondsToSelector:@selector(learningParameterForNeuralNetwork:)])
        learningParam = [self.trainingDelegate learningParameterForNeuralNetwork:self];

    if ([self.trainingDelegate respondsToSelector:@selector(neuralNetworkDidFinishLoadingTrainingExamples:)])
        [self.trainingDelegate neuralNetworkDidFinishLoadingTrainingExamples:self];
}

#pragma mark - Public Methods

- (void)loadDelegateData
{
    if (self.delegate)
        [self setupNetwork];

    if (self.trainingDelegate)
        [self loadTrainingSet];
}

- (void)trainNetwork
{
    if (!self.trainingDelegate) return;
    
    [self gradientDescent];
}

- (NSInteger)classifyInput:(float *)input
{
    float **a = (float **)calloc(L, sizeof(float *));
    a[0] = input;
    for (int l=1; l<L; l++) {
        a[l] = forwardPropagate(a[l-1], self.weights, l-1);
    }

    float max = -FLT_MAX;
    NSInteger maxIndex = 0;

    for (NSInteger k=0; k<self.weights.layerUnits[L-1]; k++) {
        float value = a[L-1][k];
        if (value > max) {
            max = value;
            maxIndex = k;
        }
    }

    return maxIndex;
}

#pragma mark - Private Mehtods

- (NSDictionary *)costFunction
{
    float J = 0.0;
    TCTheta *thetaGrad = [[TCTheta alloc] initWithLayers:self.weights.layers];

    for (int n=0; n<m; n++) {

        // Forward Propagate
        float **a = (float **)calloc(L, sizeof(float *));

        a[0] = X[n];
        for (int l=1; l<L; l++) {
            a[l] = forwardPropagate(a[l-1], self.weights, l-1);
        }

        // Calculate Cost J
        NSInteger *temp = labelArray(y[n], numLabels);

        for (int k=0; k<numLabels; k++) {
            J += -temp[k]*logf(a[L-1][k]) - (1.0 - temp[k])*logf(1.0 - a[L-1][k]);
        }

        // Backpropagate
        float **delta = (float **)calloc(L-1, sizeof(float *));

        float *delta_L = (float *)calloc(numLabels, sizeof(float));
        for (int k=0; k<numLabels; k++)
            delta_L[k] = a[L-1][k] - temp[k];

        delta[L-2] = delta_L;

        for (int l=L-2; l>0; l--) {
            delta[l-1] = computeDelta(a[l], delta[l], self.weights, l);
        }

        // Update Gradient
        for (int l=0; l<L-1; l++) {
            updateThetaGradient(&thetaGrad, l, a[l], delta[l]);
        }

        // Free memory
        for (int i=0; i<L; i++) {
            if (i != 0) {
                free(a[i]);
            }
            if (i < L-1) {
                free(delta[i]);
            }
        }
        free(delta);
        free(a);
        free(temp);
    }

    // Regularization of J
    float sum = 0.0;
    for (int l=0; l<L-1; l++) {

        float reg = 0.0;
        TCDimension WDim = [self.weights dimensionForLayer:l];
        for (NSInteger i=0; i<WDim.rows; i++) {
            for (NSInteger j=1; j<WDim.cols; j++) {
                float value = [self.weights weightValueForIndex:tci(l, i, j)];
                reg += powf(value, 2.0);
            }
        }
        sum += reg;
    }

    J *= (1.0/m);
    J += (lambda/(2.0*m)) * sum;

    // Regularization of Theta Gradient
    [thetaGrad mapToIndices:^(TCIndex index) {

        float value = [thetaGrad weightValueForIndex:index];
        value *= (1.0/m);
        if (index.j >= 1) {
            value += (lambda/m) * [self.weights weightValueForIndex:index];
        }

        [thetaGrad setWeightValue:value forIndex:index];
    }];
    
    NSDictionary *returnDict = @{@"J": @(J), @"grad" : thetaGrad};
    return returnDict;
}

- (void)gradientDescent
{
    float oldJ = FLT_MAX;

    BOOL feedback = [self.trainingDelegate respondsToSelector:@selector(neuralNetwork:didCompleteTrainingEpoch:withCost:)];

    for (int i=0; i<=maxIterations; i++) {

        NSDictionary *costFun = [self costFunction];
        TCTheta *grad = costFun[@"grad"];
        float scale = -1.0 * learningParam;
        
        [self.weights addTheta:grad multipliedByScalar:scale];

        float newJ = [costFun[@"J"] floatValue];
        if (feedback) {
            [self.trainingDelegate neuralNetwork:self didCompleteTrainingEpoch:i withCost:newJ];
        }

        if (fabs(newJ - oldJ) <= stopEpsilon)
            break;
        oldJ = newJ;
    }

    if ([self.trainingDelegate respondsToSelector:@selector(neuralNetworkDidFinishTraining:)]) {
        [self.trainingDelegate neuralNetworkDidFinishTraining:self];
    }
}

@end
