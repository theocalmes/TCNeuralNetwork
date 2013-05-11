//
//  TCStructs.h
//  NeuralNetwork
//
//  Created by theo on 4/12/13.
//  Copyright (c) 2013 Theodore Calmes. All rights reserved.
//

#define tci(l,i,j) TCIndexMake(l,i,j)

double randomRange(double low, double high)
{
    return ((double)arc4random() / 0x100000000) * (high - low) + low;
}

typedef struct TCDimension
{
    NSInteger rows;
    NSInteger cols;
} TCDimension;

static TCDimension DimensionMake(int rows, int cols)
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
