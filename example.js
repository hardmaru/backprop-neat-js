/*

Relative Clean Example Usage of neat.js

run with node.js in the command prompt:

>> node example.js

*/


var R = require('./ml/recurrent.js');
var N = require('./ml/neat.js');

var DataSet = require('./dataset.js'); // loads the dataset generator, also loads ml/neat.js and ml/recurrent.js

/*
0 - circle
1 - xor
2 - 2 gaussians
3 - spiral
*/
DataSet.generateRandomData(3); // this generates a set of dummy data.  refer to dataset.js

// settings
var penaltyConnectionFactor = 0.03;
var learnRate = 0.01;

// globals
var trainer; // NEAT trainer object to be initalised later from initModel()
var genome; // keep one genome around for good use.
var solver = new R.Solver(); // the Solver uses RMSProp

var fitnessFunc = function(genome, _backpropMode, _nCycles) {
  /*
  an example of a fitness function for NEAT for a given genome.
  this function is called inside the NEATTrainer as well for performing GA on its population
  it can handle backprop if 2nd and 3rd params are used.
  if used, this function backprops the genome over nCycles on the dataset.
  the function returns final fitness, which is based on the data fitting error
  the more negative the fitness, the crappier the given genome
  */
  "use strict";

  var backpropMode = false;
  var nCycles = 1;
  if (_backpropMode === true) backpropMode = true;
  if (typeof _nCycles !== 'undefined') nCycles = _nCycles;

  var nTrainSize = DataSet.getTrainLength();
  var trainData = DataSet.getTrainData();
  var trainLabel = DataSet.getTrainLabel();
  var nBatchSize = DataSet.getBatchLength();
  var batchData;
  var batchLabel;

  function findAvgError(genome, data, label, size, backprop) {
    // finds avg logistic regression error over a set
    var t; // ground truth label
    var e; // error
    var y; // output prediction
    var avgError = 0.0;
    genome.setupModel(size); // setup the model so recurrent.js can use it.
    genome.setInput(data); // put the input data into the network
    var G = new R.Graph(backprop); // setup the recurrent.js graph. if no backprop, faster.
    genome.forward(G); // propagates the network forward.
    var output = genome.getOutput(); // get the output from the network
    output[0] = G.sigmoid(output[0]); // in addition, take the sigmoid of the output, so output is in [0, 1]

    for (var i=0;i<size;i++) { // loops through each of the output predictions to calculate logistic error
      y = output[0].w[i]; // prediction (not to be confused w/ coordinate)
      t = label.get(i, 0); // ground truth label
      e = -(t*Math.log(y)+(1-t)*Math.log(1-y)); // logistic regression
      if (backprop) output[0].dw[i] = (y-t) / size; // for backprop
      avgError += e; // accumulate the error
    }
    avgError /= size; // make the accumulated value the average
    if (backprop === true) {
      G.backward(); // backprops the network
      solver.step(genome.model.connections, learnRate, 0.001, 5.0); // runs rmsprop with regularised weights and gradient clipping.
      genome.updateModelWeights(); // after rmsprop, have to update all the weights of the connections of this genome
    }
    return avgError;
  }

  // find the error first with no backprop
  var initError = findAvgError(genome, trainData, trainLabel, nTrainSize, false);
  var avgError = initError;
  var finalError = initError;
  var veryInitError = initError; // store a copy, in case stuff messes up later.

  if (backpropMode === true) {

    var genomeBackup = genome.copy(); // make a copy, in case backprop messes shit up.
    var origGenomeBackup = genome.copy(); // make a copy, in case backprop REALLY messes shit up.

    for (var j=0;j<nCycles;j++) { // backprops the network nCycles times, but only use a minibatch each time for performance.
      DataSet.generateMiniBatch(); // generates a random minibatch from the training data
      batchData = DataSet.getBatchData();
      batchLabel = DataSet.getBatchLabel();
      avgError = findAvgError(genome, batchData, batchLabel, nBatchSize, true); // backprop is performed, so d_avgError/d_W is stored
      if ((j > 0 && (j+1) % 20 === 0)) { // early stopping optimisations.
        finalError = findAvgError(genome, trainData, trainLabel, nTrainSize, false); // every 20 steps, see if we actually improve on entire training set
        if (finalError > initError) { // if there's no improvement after 20 steps
          genome.copyFrom(genomeBackup); // copy the genome back from the previous best genome and use that one
          break;
        } else { // if there is an improvement after 20 steps
          initError = finalError; // record current error as 'best' error
          genomeBackup = genome.copy(); // record current genome as 'best' genome
        }
      }

    }
    avgError = findAvgError(genome, trainData, trainLabel, nTrainSize, false); // at the end of it all, calculate the error of the entire training set again
    if (avgError > veryInitError) { // do one more sanity test, and compares with the very original error before the whole process
      avgError = veryInitError;
      genome.copyFrom(origGenomeBackup);
      // console.log('backprop was useless.');
    }

  }

  var penaltyConnection = genome.connections.length;
  var penaltyFactor = 1+penaltyConnectionFactor*Math.sqrt(penaltyConnection); // punish the fitness if there are lots of nodes
  // returns the fitness based on the regression error and connection penalty.
  // a more negative fitness means the given genome is crappier.
  return -avgError*penaltyFactor;
};

var backprop = function(nCycle) {
  var f = function(g) {
    // defines a fitness function that wraps the existing fitness function defined above with backprop turned on
    // since by default, the original fitness function has backprop turned off for the purpose of performance.
    return fitnessFunc(g, true, nCycle);
  };
  trainer.applyFitnessFunc(f); // this calculates the fitness for each genome, as well as backpropping them.
};

var printPerformanceMetrics = function(genome, detailMode_) {
  // calculates the accuracy for genome to predict both training and testing datasets, and prints the results
  var detailMode = false; // if true, dumps out every single datapoint and prediction.
  if (typeof detailMode_ !== 'undefined') detailMode = detailMode_;

  // helper function to use recurrent.js to calculate predictions
  function buildPredictionList(pList, thedata, thelabel, g, quantisation_) {
    // builds a list of predictions (pList), given groundtruth data, label, genome.
    // if quantisation is true, it rounds the prediction list to either 0 or 1
    "use strict";
    var i, n, y;
    var G, output;
    var acc = 0;
    var quantisation = typeof quantisation_ === 'undefined'? false : quantisation_;
    n=pList.length;
    g.setupModel(n);
    g.setInput(thedata);
    G = new R.Graph(false); // no backprop
    g.forward(G);
    output = g.getOutput();
    output[0] = G.sigmoid(output[0]);
    for (i=0;i<n;i++) {
      y = output[0].w[i]; // prediction (not to be confused w/ coordinate)
      acc += Math.round(y) === thelabel.get(i, 0)? 1 : 0;
      if (quantisation === true) {
        pList[i] = (y > 0.5)? 1.0: 0.0;
      } else {
        pList[i] = y;
      }
    }
    acc /= n;
    return acc;
  }

  function printDetail(pList, thedata, thelabel) {
    var i, n=pList.length;
    console.log("x0\tx1\tlabel\tpredict");
    for (i=0;i<n;i++) {
      console.log(thedata.get(i, 0).toPrecision(2)+'\t'+thedata.get(i, 1).toPrecision(2)+'\t'+thelabel.get(i, 0)+'\t'+pList[i].toPrecision(2));
    }
  }

  var predictionTrainList = R.zeros(DataSet.getTrainLength());
  var predictionTestList = R.zeros(DataSet.getTestLength());

  trainAccuracy = buildPredictionList(predictionTrainList, DataSet.getTrainData(), DataSet.getTrainLabel(), genome);
  testAccuracy = buildPredictionList(predictionTestList, DataSet.getTestData(), DataSet.getTestLabel(), genome);

  console.log("gen: "+N.getNumGeneration()+", fitness: "+genome.fitness.toPrecision(3)+", train accuracy: "+trainAccuracy.toPrecision(3)+", test accuracy: "+testAccuracy.toPrecision(3)+", nodes: "+genome.getNodesInUse().length+", connections: "+genome.connections.length);

  if (detailMode) {
    console.log("train set breakdown:");
    printDetail(predictionTrainList, DataSet.getTrainData(), DataSet.getTrainLabel());
    console.log("test set breakdown:");
    printDetail(predictionTestList, DataSet.getTestData(), DataSet.getTestLabel());
  }

};

var initModel = function() {
  // setup NEAT universe:
  N.init({nInput: 2, nOutput: 1, // 2 inputs (x, y) coordinate, one output (class)
    initConfig: "all", // initially, each input is connected to each output when "all" is used
    activations : "default", // [SIGMOID, TANH, RELU, GAUSSIAN, SIN, ABS, MULT, SQUARE, ADD] for "default"
  });
  // setup NEAT trainer with the hyper parameters for GA.
  trainer = new N.NEATTrainer({
    new_node_rate : 0.2, // probability of a new node created for each genome during each evolution cycle
    new_connection_rate : 0.5, // probability of a new connection created for each genome during each evolution cycle, if it can be created
    num_populations: 5, // cluster the population into 5 sub populations that are similar using k-medoids
    sub_population_size : 20, // each sub population has 20 members, so 100 genomes in total
    init_weight_magnitude : 0.25, // randomise initial weights to be gaussians with zero mean, and this stdev.
    mutation_rate : 0.9, // probability of mutation for weights (for this example i made it large)
    mutation_size : 0.005, // if weights are mutated, how much we mutate them by in stdev? (I made it very small for this example)
    extinction_rate : 0.5, // probably that the worst performing sub population goes extinct at each evolution cycle
  }); // the initial population of genomes is randomly created after N.NEATTrainer constructor is called.
  trainer.applyFitnessFunc(fitnessFunc); // this would calculate the fitness for each genome in the population, and clusters them into the 5 sub populations
};

/*
Main part of the code.
*/
initModel();
genome = trainer.getBestGenome();

for (var i = 0; i < 10; i++) { // evolve and backprop 10 times
  printPerformanceMetrics(genome); // print out the performance metrics
  trainer.evolve();
  backprop(600);
  genome = trainer.getBestGenome();
}

printPerformanceMetrics(genome, true); // print out the final performance metrics with more details

/*
if you want to export the genome and save it for future use, you can do the following:
*/
genome = trainer.getBestGenome(); // get best genome in the trainer's population after training

var data_string = genome.toJSON() // dump the genome to json format

console.log('json of best genome:');
console.log(data_string);
// save json data string somewhere

// in some other app, declare a genome from N.Genome() class, and load the json data, like:
// genome.fromJSON(data_string);
// so the genome can be used to predict things.


