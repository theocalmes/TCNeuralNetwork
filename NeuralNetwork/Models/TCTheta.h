//
//  TCTheta.h
//  NeuralNetwork
//
//  Created by theo on 4/5/13.
//  Copyright (c) 2013 Theodore Calmes. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "TCStructs.h"

@interface TCTheta : NSObject <NSCoding>

@property (assign, readonly, nonatomic) float **weights;
@property (strong, readonly, nonatomic) NSArray *layers;
@property (assign, readonly, nonatomic) NSInteger *layerUnits;

- (id)initWithLayers:(NSArray *)layers;
- (void)randomizeValuesWithEpsilon:(float)epsilon;

- (void)setWeightValue:(float)value forIndex:(TCIndex)index;
- (float)weightValueForIndex:(TCIndex)index;

- (void)addMatrix:(float *)matrix toLayer:(NSInteger)l;
- (void)addTheta:(TCTheta *)theta;
- (void)addTheta:(TCTheta *)theta multipliedByScalar:(float)alpha;

- (TCDimension)dimensionForLayer:(NSInteger)layer;

- (void)mapToIndices:(void(^)(TCIndex))block;

- (void)printWeightMatrix:(NSInteger)l;

@end
