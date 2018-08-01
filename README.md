# MultiLayerRNN_HawkesProcess
Trains a multi-layer recurrent neural net work with simulated time-dependent data

Background:
A self-exciting event is something where one occurence of an event is more likely 
to trigger repeated occurences of that same event. For example, an event like an earthquake is likely 
to trigger succesive earthquakes (aftershocks) in the span of time immediately following the initial earthquake. As the time since the last earthquake increases, the probablility of another earthquake occuring declines. Events of this nature are known as Hawkes Processes.

The functions in this repository build and train a multi-layer RNN to learn the elasped time between event 
occurences, known as inter-arrival times, in a Hawkes Process using statiscially 
simulated data for training. 

The run.lua script establishes default learning parameters and calls the train_1dim_hawkes.lua to train the model.
