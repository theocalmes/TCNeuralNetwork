//
//  TCNeuralNetwork.m
//  NeuralNetwork
//
//  Created by Theodore Calmes on 11/18/12.
//  Copyright (c) 2012 Theodore Calmes. All rights reserved.
//

#import "NeuralNetwork.h"

#pragma mark - Computation

float sigmoid(float z)
{
    return 1.0/(1.0 + expf(-z));
}

float gradientActivity(float a)
{
    return a*(1.0-a);
}

float *forwardPropagate(float *al, TCTheta *theta, NSInteger l)
{
    TCDimension WlDim = [theta dimensionForLayer:l];
    NSInteger M = WlDim.rows;
    NSInteger N = WlDim.cols;

    float *a = (float *)calloc(N, sizeof(float));
    a[0] = 1.0;
    vDSP_vadd(&a[1], 1, al, 1, &a[1], 1, N-1);

    float *next = malloc(M * sizeof(float));
    vDSP_mmul(theta.weights[l], 1, a, 1, next, 1, M, 1, N);

    for (NSInteger i = 0; i < M; ++i)
        next[i] = sigmoid(next[i]);

    free(a);

    return next;
}

float *computeDelta(float *a, float *nextDelta, TCTheta *theta, NSInteger l)
{
    TCDimension WlDim = [theta dimensionForLayer:l];
    NSInteger M = WlDim.rows;
    NSInteger N = WlDim.cols;

    float *transposed = malloc(N * M * sizeof(float));
    vDSP_mtrans(theta.weights[l], 1, transposed, 1, N, M);

    float *product = malloc(N * sizeof(float));
    vDSP_mmul(transposed, 1, nextDelta, 1, product, 1, N, 1, M);

    float *delta = malloc((N-1) * sizeof(float));

    for (NSInteger i = 1; i < N; ++i)
        delta[i-1] = product[i] * gradientActivity(a[i-1]);

    free(transposed);
    free(product);

    return delta;
}

void updateThetaGradient(TCTheta **theta, NSInteger l, float *al, float *deltaNext)
{
    TCDimension WlDim = [*theta dimensionForLayer:l];
    NSInteger M = WlDim.rows;
    NSInteger N = WlDim.cols;

    float *a = (float *)calloc(N, sizeof(float));
    a[0] = 1.0;
    vDSP_vadd(&a[1], 1, al, 1, &a[1], 1, N-1);

    float *product = (float *)calloc(M * N, sizeof(float));
    vDSP_mmul(deltaNext, 1, a, 1, product, 1, M, N, 1);

    free(a);
    
    [*theta addMatrix:product toLayer:l];
}

float *labelArray(float label, NSInteger total)
{
    float *array = (float *)calloc(total, sizeof(float));
    array[(NSInteger)label] = 1;
    return array;
}

#pragma mark - Init

@interface TCFeedForwardNetwork ()
@property (strong, readwrite, nonatomic) TCTheta *weights;
@end

@implementation TCFeedForwardNetwork
{
    float **X;
    float *y;

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
    self = [super init];
    if (self) {
        _weights = [[TCTheta alloc] initWithLayers:neuronLayers];
        [_weights randomizeValuesWithRange:TCRangeMake(-0.12, 0.12)];

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

- (float)classifyInput:(float *)input
{
    float **a = (float **)calloc(L, sizeof(float *));
    a[0] = input;
    for (NSInteger l = 1; l < L; l++) {
        a[l] = forwardPropagate(a[l-1], self.weights, l-1);
    }

    if (self.weights.layerUnits[L-1] == 1) {
        float returnValue = a[L-1][0];
        for (NSInteger i = 1; i < L; i++) {
            free(a[i]);
        }
        free(a);

        return returnValue;
    }
    
    float max = -FLT_MAX;
    NSInteger maxIndex = 0;

    for (NSInteger k = 0; k < self.weights.layerUnits[L-1]; k++) {
        float value = a[L-1][k];
        if (value > max) {
            max = value;
            maxIndex = k;
        }
    }

    for (NSInteger i = 1; i < L; i++) {
        free(a[i]);
    }
    free(a);

    return maxIndex;
}

#pragma mark - Private Mehtods

- (NSDictionary *)costFunction
{
    float J = 0.0;
    TCTheta *thetaGrad = [[TCTheta alloc] initWithLayers:self.weights.layers];

    for (NSInteger n = 0; n < m; n++) {

        // Forward Propagate
        float **a = (float **)calloc(L, sizeof(float *));

        a[0] = X[n];
        for (NSInteger l = 1; l < L; l++) {
            a[l] = forwardPropagate(a[l-1], self.weights, l-1);
        }

        // Calculate Cost J
        float *temp;
        if (numLabels > 1) {
            temp = labelArray(y[n], numLabels);
        }
        else {
            temp = (float *)calloc(1, sizeof(float));
            temp[0] = y[n];
        }

        for (NSInteger k = 0; k < numLabels; k++) {
            J += -temp[k]*logf(a[L-1][k]) - (1.0 - temp[k])*logf(1.0 - a[L-1][k]);
        }

        // Backpropagate
        float **delta = (float **)calloc(L-1, sizeof(float *));

        float *delta_L = (float *)calloc(numLabels, sizeof(float));
        for (NSInteger k = 0; k < numLabels; k++)
            delta_L[k] = a[L-1][k] - temp[k];

        delta[L-2] = delta_L;

        for (NSInteger l = L-2; l > 0; l--) {
            delta[l-1] = computeDelta(a[l], delta[l], self.weights, l);
        }

        // Update Gradient
        for (NSInteger l = 0; l < L-1; l++) {
            updateThetaGradient(&thetaGrad, l, a[l], delta[l]);
        }

        // Free memory
        for (NSInteger i = 0; i < L; i++) {
            if (i != 0) {
                free(a[i]);
            }
            if (i < L-1) {
                free(delta[i]);
            }
        }
        free(temp);
        free(delta);
        free(a);
    }

    // Regularization of Theta Gradient and Cost

    float **dW = thetaGrad.weights;
    float **W = _weights.weights;

    float sum = 0.0;
    for (NSInteger l = 0; l < L-1; l++) {

        TCDimension WDim = [thetaGrad dimensionForLayer:l];
        NSInteger size = WDim.rows * WDim.cols;

        float *tempW = malloc(size * sizeof(float));

        float regScale = lambda/m;
        float avg = 1.0/m;

        vDSP_vsmul(W[l],1, &regScale, tempW, 1, size);
        vDSP_vsmul(dW[l], 1, &avg, dW[l], 1, size);

        vDSP_vadd(dW[l], 1, tempW, 1, dW[l], 1, size);

        float reg = 0.0;
        vDSP_dotpr(W[l], 1, W[l], 1, &reg, size);

        for (NSInteger i = 0; i < WDim.rows; i++) {
            dW[l][i * WDim.cols] -= tempW[i * WDim.cols];
            reg -= powf(W[l][i * WDim.cols], 2);
        }

        free(tempW);

        sum += reg;
    }
    thetaGrad.weights = dW;

    J *= (1.0/m);
    J += (lambda/(2.0*m)) * sum;

    NSDictionary *returnDict = @{@"J": @(J), @"grad" : thetaGrad};
    return returnDict;
}

- (void)gradientDescent
{
    float oldJ = FLT_MAX;

    BOOL feedback = [self.trainingDelegate respondsToSelector:@selector(neuralNetwork:didCompleteTrainingEpoch:withCost:)];

    for (NSInteger i = 0; i <= maxIterations; i++) {

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
