//
//  Functions.h
//  NeuralNetwork
//
//  Created by theo on 5/11/13.
//  Copyright (c) 2013 Theodore Calmes. All rights reserved.
//

#import "TCStructs.h"

static double randomRange(TCRandomRange range)
{
    return ((double)arc4random() / 0x100000000) * (range.high - range.low) + range.low;
}