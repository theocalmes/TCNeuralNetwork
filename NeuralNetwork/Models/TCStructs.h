//
//  TCStructs.h
//  NeuralNetwork
//
//  Created by theo on 4/12/13.
//  Copyright (c) 2013 Theodore Calmes. All rights reserved.
//

#define tci(l,i,j) TCIndexMake(l,i,j)

#define tci(l,i,j) TCIndexMake(l,i,j)

typedef struct TCRandomRange
{
    float low;
    float high;
} TCRandomRange;

static TCRandomRange TCRandomRangeMake(float low, float high)
{
    TCRandomRange range; range.low = low; range.high = high; return range;
}

typedef struct TCDimension
{
    NSInteger rows;
    NSInteger cols;
} TCDimension;

static TCDimension TCDimensionMake(int rows, int cols)
{
    TCDimension dim; dim.rows = rows; dim.cols = cols; return dim;
}

typedef struct TCIndex {
    NSInteger l;
    NSInteger i;
    NSInteger j;
} TCIndex;

static TCIndex TCIndexMake(NSInteger l, NSInteger i, NSInteger j)
{
    TCIndex index;
    index.l = l;
    index.i = i;
    index.j = j;

    return index;
}
