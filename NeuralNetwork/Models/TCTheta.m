//
//  TCTheta.m
//  NeuralNetwork
//
//  Created by theo on 4/5/13.
//  Copyright (c) 2013 Theodore Calmes. All rights reserved.
//

#import "NeuralNetwork.h"

TCDimension layerDimention(NSInteger unitsInLayer, NSInteger unitsInNextLayer)
{
    return DimensionMake(unitsInNextLayer, unitsInLayer + 1);
}

@interface TCTheta ()
@end

@implementation TCTheta

- (id)initWithLayers:(NSArray *)layers
{
    self = [super init];
    if (!self) return nil;

    _layers = layers;

    _weights = (float **)calloc(layers.count - 1, sizeof(float *));
    _layerUnits = (NSInteger)calloc(layers.count, sizeof(NSInteger));
    
    for (NSInteger l = 0; l < layers.count - 1; l++) {

        _layerUnits[l] = [layers[l] integerValue];
        
        TCDimension WlDim = layerDimention([layers[l] integerValue], [layers[l+1] integerValue]);

        _weights[l] = (float *)calloc(WlDim.cols * WlDim.rows, sizeof(float));
    }
    _layerUnits[layers.count - 1] = [layers.lastObject integerValue];

    return self;
}

- (TCDimension)dimensionForLayer:(NSInteger)layer
{
    if (layer >= self.layers.count) return DimensionMake(0, 0);

    return layerDimention(self.layerUnits[layer], self.layerUnits[layer + 1]);
}

- (void)randomizeValuesWithEpsilon:(float)epsilon
{
    [self mapToIndices:^(TCIndex index) {
        float rand = (float)randomRange(-epsilon, epsilon);
        [self setWeightValue:rand forIndex:index];
    }];
}

#pragma mark - Setters and Getters

- (void)setWeightValue:(float)value forIndex:(TCIndex)index
{
    TCDimension WlDim = layerDimention(self.layerUnits[index.l], self.layerUnits[index.l + 1]);
    self.weights[index.l][index.i * WlDim.cols + index.j] = value;
}

- (float)weightValueForIndex:(TCIndex)index
{
    TCDimension WlDim = layerDimention(self.layerUnits[index.l], self.layerUnits[index.l + 1]);
    return self.weights[index.l][index.i * WlDim.cols + index.j];
}

#pragma mark - Matrix Math

- (void)addMatrix:(float *)matrix scaleBy:(float)scale toLayer:(NSInteger)l
{
    TCDimension WlDim = layerDimention(self.layerUnits[l], self.layerUnits[l + 1]);
    NSInteger length = WlDim.cols * WlDim.rows;

    float *scaled __attribute__ ((aligned)) = malloc(sizeof(float) * length);
    vDSP_vsmul(matrix, 1, &scale, scaled, 1, length);

    vDSP_vadd(_weights[l], 1, scaled, 1, _weights[l], 1, length);

    free(scaled);
}

- (void)addMatrix:(float *)matrix toLayer:(NSInteger)l
{
    TCDimension WlDim = layerDimention(self.layerUnits[l], self.layerUnits[l + 1]);
    NSInteger length = WlDim.cols * WlDim.rows;

    vDSP_vadd(_weights[l], 1, matrix, 1, _weights[l], 1, length);

    free(matrix);
}

- (void)addTheta:(TCTheta *)theta
{
    for (NSInteger l = 0; l < self.layers.count-1; l++) {
        [self addMatrix:theta.weights[l] toLayer:l];
    }
}
- (void)addTheta:(TCTheta *)theta multipliedByScalar:(float)alpha
{
    for (NSInteger l = 0; l < self.layers.count-1; l++) {
        [self addMatrix:theta.weights[l] scaleBy:alpha toLayer:l];
    }
}

#pragma mark - Index Mapping

- (void)mapToIndices:(void (^)(TCIndex))block
{
    for (NSInteger l = 0; l < self.layers.count-1; l++) {
        TCDimension WlDim = layerDimention([self.layers[l] integerValue], [self.layers[l+1] integerValue]);
        for (NSInteger i = 0; i < WlDim.rows; i++) {
            for (NSInteger j = 0; j < WlDim.cols; j++) {
                block(tci(l, i, j));
            }
        }
    }
}

#pragma mark - Logging

- (void)printWeightMatrix:(NSInteger)l
{
    TCDimension WlDim = [self dimensionForLayer:l];
    float *Wl = self.weights[l];

    printf("\nPrinting Weight[%d]:\n", l);
    for (NSInteger i = 0; i < WlDim.rows; i++) {
        for (NSInteger j = 0; j < WlDim.cols; j++) {
            printf(" %e ", Wl[i*WlDim.cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

#pragma mark - Memory Management

- (void)dealloc
{
    for (NSInteger l = 0; l < _layers.count-1; l++) {
        free(_weights[l]);
    }
    free(_weights);
    free(_layerUnits);
}

#pragma mark - NSCoding

- (id)initWithCoder:(NSCoder *)aDecoder
{
    self = [super init];
    if (!self) return nil;

    _layers = [aDecoder decodeObjectForKey:@"layers"];
    
    NSData *layerUnitsData = [aDecoder decodeObjectForKey:@"layerUnits"];
    NSInteger *temp = (NSInteger *)[layerUnitsData bytes];
    
    _layerUnits = (NSInteger *)calloc(_layers.count, sizeof(NSInteger));
    for(unsigned char l = 0; l < _layers.count; l++) {
        _layerUnits[l] = temp[l];
    }

    _weights = (float **)calloc(_layers.count - 1, sizeof(float *));
    for (NSInteger l = 0; l < _layers.count-1; l++) {
        NSString *key = [NSString stringWithFormat:@"w_%d", l];
        TCDimension WlDim = [self dimensionForLayer:l];

        float *Wl = (float *)calloc(WlDim.rows * WlDim.cols, sizeof(float));

        NSData *WlData = [aDecoder decodeObjectForKey:key];
        float *temp = (float *)[WlData bytes];

        for(NSInteger k = 0; k < WlDim.rows*WlDim.cols; k++) {
            Wl[k] = temp[k];
        }

        _weights[l] = Wl;
    }

    return self;
}

- (void)encodeWithCoder:(NSCoder *)aCoder
{
    [aCoder encodeObject:self.layers forKey:@"layers"];
    [aCoder encodeObject:[NSData dataWithBytes:(void *)self.layerUnits length:self.layers.count*sizeof(NSInteger)] forKey:@"layerUnits"];

    for (NSInteger l = 0; l < self.layers.count-1; l++) {
        NSString *key = [NSString stringWithFormat:@"w_%d", l];
        TCDimension WlDim = [self dimensionForLayer:l];
        NSInteger length = WlDim.cols * WlDim.rows * sizeof(float);
        [aCoder encodeObject:[NSData dataWithBytes:(void *)self.weights[l] length:length] forKey:key];
    }
}

@end
