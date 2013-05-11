//
//  TCNeuralNetworkDelegate.h
//  NeuralNetwork
//
//  Created by theo on 4/12/13.
//  Copyright (c) 2013 Theodore Calmes. All rights reserved.
//

#import <Foundation/Foundation.h>

@class TCTheta;
@class TCFeedForwardNetwork;

/** The TCFeedForwardNetworkDelegate protocol provides you with a way to load weights for the network to use. If you implement this function and it returns a nil value, the network will go ahead and initialize itself using random initial values.
 */
@protocol TCFeedForwardNetworkDelegate <NSObject>

@optional

/** Asks the delegate for a Theta object for the network to initialize itself with.

 Note: loading takes place when calling loadDelegateData from the TCFeedForwardNetwork object.
 
 @param network An object representing the neural network request this information.
 @return The weights for the network to initialize with. Default value is random weights.
 */
- (TCTheta *)weightsForNeuralNetwork:(TCFeedForwardNetwork *)network;

/** Gets called when the weights are loaded.

 Note: this method will only be called if you implement weightsForNeuralNetwork: and have it return a non nil value.

 @param network An object representing the neural network whose weights were just loaded.
 */
- (void)weightsDidLoadForNeuralNetwork:(TCFeedForwardNetwork *)network;

@end
