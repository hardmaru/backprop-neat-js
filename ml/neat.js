/*globals paper, console, $ */
/*jslint nomen: true, undef: true, sloppy: true */

// NEAT implementation

/*

@licstart  The following is the entire license notice for the
JavaScript code in this page.

Copyright (C) 2016 david ha, otoro.net, otoro labs

MIT License.

@licend  The above is the entire license notice
for the JavaScript code in this page.
*/

// implementation of neat algorithm with recurrent.js graphs
// code is not modular or oop, ad done in a fortran/scientific computing style
// apologies in advance if that ain't ur taste.
if (typeof module != "undefined") {
  var R = require('./recurrent.js');
  var kmedoids = require('./kmedoids.js');
}

var N = {};

(function(global) {
  "use strict";

  // constants
  var NODE_INPUT = 0;
  var NODE_OUTPUT = 1;
  var NODE_BIAS = 2;
  // hidden layers
  var NODE_SIGMOID = 3;
  var NODE_TANH = 4;
  var NODE_RELU = 5;
  var NODE_GAUSSIAN = 6;
  var NODE_SIN = 7;
  var NODE_COS = 8;
  var NODE_ABS = 9;
  var NODE_MULT = 10;
  var NODE_ADD = 11;
  var NODE_MGAUSSIAN = 12; // multi-dim gaussian (do gaussian to each input then mult)
  var NODE_SQUARE = 13;

  var NODE_INIT = NODE_ADD;

  var MAX_TICK = 100;

  var operators = [null, null, null, 'sigmoid', 'tanh', 'relu', 'gaussian', 'sin', 'cos', 'abs', 'mult', 'add', 'mult', 'add'];

  // for connections
  var IDX_CONNECTION = 0;
  var IDX_WEIGHT = 1;
  var IDX_ACTIVE = 2;

  var debug_mode = false; // shuts up the messages.

  // for initialisation ("none" means initially just 1 random connection, "one" means 1 node that connects all, "all" means fully connected)
  var initConfig = "none"; // default for generating images.

  //var activations = [NODE_SIGMOID, NODE_TANH, NODE_RELU, NODE_GAUSSIAN, NODE_SIN, NODE_COS, NODE_MULT, NODE_ABS, NODE_ADD, NODE_MGAUSSIAN, NODE_SQUARE];
  //var activations = [NODE_SIGMOID, NODE_TANH, NODE_RELU, NODE_GAUSSIAN, NODE_SIN, NODE_COS, NODE_ADD];
  // var activations = [NODE_SIGMOID, NODE_TANH, NODE_RELU, NODE_GAUSSIAN, NODE_SIN, NODE_COS, NODE_MULT, NODE_ABS, NODE_ADD, NODE_SQUARE];

  // keep below for generating images, for the default.
  var activations_default = [NODE_SIGMOID, NODE_TANH, NODE_RELU, NODE_GAUSSIAN, NODE_SIN, NODE_ABS, NODE_MULT, NODE_SQUARE, NODE_ADD];
  var activations_all = [NODE_SIGMOID, NODE_TANH, NODE_RELU, NODE_GAUSSIAN, NODE_SIN, NODE_COS, NODE_MULT, NODE_ABS, NODE_ADD, NODE_MGAUSSIAN, NODE_SQUARE];
  var activations_minimal = [NODE_RELU, NODE_TANH, NODE_GAUSSIAN, NODE_ADD];

  var activations = activations_default;

  var getRandomActivation = function() {
    var ix = R.randi(0, activations.length);
    return activations[ix];
  };

  var gid = 0;
  var getGID = function() {
    var result = gid;
    gid += 1;
    return result;
  };

  var nodes = []; // this array holds all nodes
  var connections = []; // index of connections here is the 'innovation' value

  var copyArray = function(x) {
    // returns a copy of floatArray
    var n = x.length;
    var result = new Array(n);
    for (var i=0;i<n;i++) {
      result[i]=x[i];
    }
    return result;
  };

  function copyConnections(newC) {
    var i, n;
    n = newC.length;
    var copyC = [];
    for (i=0;i<n;i++) { // connects input and bias to init dummy node
      copyC.push([newC[i][0], newC[i][1]]);
    }
    return copyC;
  }

  var getNodes = function() {
    return copyArray(nodes);
  };

  var getConnections = function() {
    return copyConnections(connections);
  };

  var nInput = 1;
  var nOutput = 1;
  var outputIndex = 2; // [bias, input, output]
  var initNodes = [];
  var initMu = 0.0, initStdev = 1.0; // randomised param initialisation.
  var mutationRate = 0.2;
  var mutationSize = 0.5;
  var generationNum = 0;

  var incrementGenerationCounter = function() { generationNum += 1; };

  function getRandomRenderMode() {
    // more chance to be 0 (1/2), and then 1/3 to be 1, and 1/6 to be 2
    var z = R.randi(0, 6);
    if (z<3) return 0;
    if (z<5) return 1;
    return 2;
  }

  var renderMode = getRandomRenderMode(); // 0 = sigmoid (1 = gaussian, 2 = tanh+abs

  var randomizeRenderMode = function() {
    renderMode = getRandomRenderMode();
    if (debug_mode) console.log('render mode = '+renderMode);
  };

  var setRenderMode = function(rMode) {
    renderMode = rMode;
  };

  var getRenderMode = function() {
    return renderMode;
  };

  function getOption(opt, index, orig) {
    if (opt && typeof opt[index] !== null) { return opt[index]; }
    return orig;
  }

  var init = function(opt) {
    var i, j;
    nInput = getOption(opt, 'nInput', nInput);
    nOutput = getOption(opt, 'nOutput', nOutput);
    initConfig = getOption(opt, 'initConfig', initConfig);
    if (typeof opt.activations !== "undefined" && opt.activations === "all") {
      activations = activations_all;
    }
    if (typeof opt.activations !== "undefined" && opt.activations === "minimal") {
      activations = activations_minimal;
      //operators[NODE_OUTPUT] = 'tanh'; // for minimal, make output be an tanh operator.
    }

    outputIndex = nInput+1; // index of output start (bias so add 1)
    nodes = [];
    connections = [];
    generationNum = 0; // initialisze the gen counter
    // initialise nodes
    for (i=0;i<nInput;i++) {
      nodes.push(NODE_INPUT);
    }
    nodes.push(NODE_BIAS);
    for (i=0;i<nOutput;i++) {
      nodes.push(NODE_OUTPUT);
    }
    // initialise connections. at beginning only connect inputs to outputs
    // initially, bias has no connections and that must be grown.

    if (initConfig === "all") {
      for (j=0;j<nOutput;j++) {
        for (i=0;i<nInput+1;i++) {
          connections.push([i, outputIndex+j]);
        }
      }
    } else if (initConfig === "one") {
      // push initial dummy node
      nodes.push(NODE_ADD);
      var dummyIndex = nodes.length-1;
      for (i=0;i<nInput+1;i++) { // connects input and bias to init dummy node
        connections.push([i, dummyIndex]);
      }
      for (i=0;i<nOutput;i++) { // connects dummy node to output
        connections.push([dummyIndex, outputIndex+i]);
      }
    }


  };

  function getNodeList(node_type) {
    // returns a list of locations (index of global node array) containing
    // where all the output nodes are
    var i, n;
    var result = [];
    for (i=0,n=nodes.length;i<n;i++) {
      if (nodes[i] === node_type) {
        result.push(i);
      }
    }
    return result;
  }

  var Genome = function(initGenome) {
    var i, j;
    var n;
    var c; // connection storage.

    this.connections = [];
    // create or copy initial connections
    if (initGenome && typeof initGenome.connections !== null) {
      for (i=0,n=initGenome.connections.length;i<n;i++) {
        this.connections.push(R.copy(initGenome.connections[i]));
      }
    } else {

      if (initConfig === "all") {
        // copy over initial connections (nInput + connectBias) * nOutput
        for (i=0,n=(nInput+1)*nOutput;i<n;i++) {
          c = R.zeros(3); // innovation number, weight, enabled (1)
          c[IDX_CONNECTION] = i;
          c[IDX_WEIGHT] = R.randn(initMu, initStdev);
          c[IDX_ACTIVE] = 1;
          this.connections.push(c);
        }
      }  else if (initConfig === "one") {

        for (i=0,n=(nInput+1)+nOutput;i<n;i++) {
          c = R.zeros(3); // innovation number, weight, enabled (1)
          c[IDX_CONNECTION] = i;
          // the below line assigns 1 to initial weights from dummy node to output
          c[IDX_WEIGHT] = (i < (nInput+1)) ? R.randn(initMu, initStdev) : 1.0;
          //c[IDX_WEIGHT] = R.randn(initMu, initStdev);
          c[IDX_ACTIVE] = 1;
          this.connections.push(c);
        }

      }

    }
  };

  Genome.prototype = {
    copy: function() {
      // makes a copy of itself and return it (returns a Genome class)
      var result = new Genome(this);
      // copies other tags if exist
      if (this.fitness) result.fitness = this.fitness;
      if (this.cluster) result.cluster = this.cluster;
      return result;
    },
    importConnections: function(cArray) {
      var i, n;
      this.connections = [];
      var temp;
      for (i=0,n=cArray.length;i<n;i++) {
        temp = new R.zeros(3);
        temp[0] = cArray[i][0];
        temp[1] = cArray[i][1];
        temp[2] = cArray[i][2];
        this.connections.push(temp);
      }
    },
    copyFrom: function(sourceGenome) {
      // copies connection weights from sourceGenome to this genome, hence making a copy
      this.importConnections(sourceGenome.connections);
      if (sourceGenome.fitness) this.fitness = sourceGenome.fitness;
      if (sourceGenome.cluster) this.cluster = sourceGenome.cluster;
    },
    mutateWeights: function(mutationRate_, mutationSize_) {
      // mutates each weight of current genome with a probability of mutationRate
      // by adding a gaussian noise of zero mean and mutationSize stdev to it
      var mRate = mutationRate_ || mutationRate;
      var mSize = mutationSize_ || mutationSize;

      var i, n;
      for (i=0,n=this.connections.length;i<n;i++) {
        if (Math.random() < mRate) {
          this.connections[i][IDX_WEIGHT] += R.randn(0, mSize);
        }
      }
    },
    areWeightsNaN: function() {
      // if any weight value is NaN, then returns true and break.
      var origWeight;
      var i, n;
      for (i=0,n=this.connections.length;i<n;i++) {
          origWeight = this.connections[i][IDX_WEIGHT];
          if (isNaN(origWeight)) {
            return true;
          }
      }
      return false;
    },
    clipWeights: function(maxWeight_) {
      // weight clipping to +/- maxWeight_
      // this function also checks if weights are NaN. if so, zero them out.
      var maxWeight = maxWeight_ || 50.0;
      maxWeight = Math.abs(maxWeight);
      var origWeight;

      var detectedNaN = false;

      var i, n;
      for (i=0,n=this.connections.length;i<n;i++) {
          origWeight = this.connections[i][IDX_WEIGHT];

          R.assert(!isNaN(origWeight), 'weight had NaN.  backprop messed shit up.');

          origWeight = Math.min(maxWeight, origWeight);
          origWeight = Math.max(-maxWeight,origWeight);

          this.connections[i][IDX_WEIGHT] = origWeight;
      }
    },
    getAllConnections: function() {
      return connections;
    },
    addRandomNode: function() {
      // adds a new random node and assigns it a random activation gate
      // if there are no connections, don't add a new node
      if (this.connections.length === 0) return;
      var c = R.randi(0, this.connections.length); // choose random connection
      // only proceed if the connection is actually active.
      if (this.connections[c][IDX_ACTIVE] !== 1) return;

      var w = this.connections[c][1];

      this.connections[c][IDX_ACTIVE] = 0; // disable the connection
      var nodeIndex = nodes.length;
      nodes.push(getRandomActivation()); // create the new node globally

      var innovationNum = this.connections[c][0];
      var fromNodeIndex = connections[innovationNum][0];
      var toNodeIndex = connections[innovationNum][1];

      var connectionIndex = connections.length;
      // make 2 new connection globally
      connections.push([fromNodeIndex, nodeIndex]);
      connections.push([nodeIndex, toNodeIndex]);

      // put in this node locally into genome
      var c1 = R.zeros(3);
      c1[IDX_CONNECTION] = connectionIndex;
      c1[IDX_WEIGHT] = 1.0; // use 1.0 as first connection weight
      c1[IDX_ACTIVE] = 1;
      var c2 = R.zeros(3);
      c2[IDX_CONNECTION] = connectionIndex+1;
      c2[IDX_WEIGHT] = w; // use old weight for 2nd connection
      c2[IDX_ACTIVE] = 1;

      this.connections.push(c1);
      this.connections.push(c2);
    },
    getNodesInUse: function() {
      var i, n, connectionIndex, nodeIndex;
      var nodesInUseFlag = R.zeros(nodes.length);
      var nodesInUse = [];
      var len = nodes.length;

      for (i=0,n=this.connections.length;i<n;i++) {
        connectionIndex = this.connections[i][0];
        nodeIndex = connections[connectionIndex][0];
        nodesInUseFlag[nodeIndex] = 1;
        nodeIndex = connections[connectionIndex][1];
        nodesInUseFlag[nodeIndex] = 1;
      }
      for (i=0,n=len;i<n;i++) {
        if (nodesInUseFlag[i] === 1 || (i < nInput+1+nOutput) ) { // if node is input, bias, output, throw it in too
          //console.log('pushing node #'+i+' as node in use');
          nodesInUse.push(i);
        }
      }
      return nodesInUse;
    },
    addRandomConnection: function() {
      // attempts to add a random connection.
      // if connection exists, then does nothing (ah well)

      var i, n, connectionIndex, nodeIndex;

      var nodesInUse = this.getNodesInUse();
      var len = nodes.length;

      //var fromNodeIndex = R.randi(0, nodes.length);
      //var toNodeIndex = R.randi(outputIndex, nodes.length); // includes bias.

      var slack = 0;
      var r1 = R.randi(0, nodesInUse.length - nOutput);
      if (r1 > nInput+1) slack = nOutput; // skip the outputs of the array.
      var fromNodeIndex = nodesInUse[r1 + slack]; // choose anything but output nodes
      var toNodeIndex = nodesInUse[R.randi(outputIndex, nodesInUse.length)]; // from output to other nodes

      var fromNodeUsed = false;
      var toNodeUsed = false;

      if (fromNodeIndex === toNodeIndex) {
        //console.log('addRandomConnection failed to connect '+fromNodeIndex+' to '+toNodeIndex);
        return; // can't be the same index.
      }

      // cannot loop back out from the output.
      /*
      if (fromNodeIndex >= outputIndex && fromNodeIndex < (outputIndex+nOutput)){
        //console.log('addRandomConnection failed to connect '+fromNodeIndex+' to '+toNodeIndex);
        return;
      }
      */

      // the below set of code will test if selected nodes are actually used in network connections
      for (i=0,n=this.connections.length;i<n;i++) {
        connectionIndex = this.connections[i][0];
        if ((connections[connectionIndex][0] === fromNodeIndex) || (connections[connectionIndex][1] === fromNodeIndex)) {
          fromNodeUsed = true; break;
        }
      }
      for (i=0,n=this.connections.length;i<n;i++) {
        connectionIndex = this.connections[i][0];
        if ((connections[connectionIndex][0] === toNodeIndex) || (connections[connectionIndex][1] === toNodeIndex)) {
          toNodeUsed = true; break;
        }
      }

      if (fromNodeIndex < nInput+1) fromNodeUsed = true; // input or bias
      if ((toNodeIndex >= outputIndex) && (toNodeIndex < outputIndex+nOutput)) toNodeUsed = true; // output

      if (!fromNodeUsed || !toNodeUsed) {
        if (debug_mode) {
          console.log('nodesInUse.length = '+nodesInUse.length);
          console.log('addRandomConnection failed to connect '+fromNodeIndex+' to '+toNodeIndex);
        }
        return; // only consider connections in current net.
      }
      //console.log('attempting to connect '+fromNodeIndex+' to '+toNodeIndex);

      var searchIndex = -1; // see if connection already exist.
      for (i=0,n=connections.length;i<n;i++) {
        if (connections[i][0] === fromNodeIndex && connections[i][1] === toNodeIndex) {
          searchIndex = i; break;
        }
      }

      if (searchIndex < 0) {
        // great, this connection doesn't exist yet!
        connectionIndex = connections.length;
        connections.push([fromNodeIndex, toNodeIndex]);

        var c = R.zeros(3); // innovation number, weight, enabled (1)
        c[IDX_CONNECTION] = connectionIndex;
        c[IDX_WEIGHT] = R.randn(initMu, initStdev);
        c[IDX_ACTIVE] = 1;
        this.connections.push(c);
      } else {
        var connectionIsInGenome = false;
        for (i=0,n=this.connections.length; i<n; i++) {
          if (this.connections[i][IDX_CONNECTION] === searchIndex) {
            // enable back the index (if not enabled)
            if (this.connections[i][IDX_ACTIVE] === 0) {
              this.connections[i][IDX_WEIGHT] = R.randn(initMu, initStdev); // assign a random weight to reactivated connections
              this.connections[i][IDX_ACTIVE] = 1;
            }
            connectionIsInGenome = true;
            break;
          }
        }
        if (!connectionIsInGenome) {
          // even though connection exists globally, it isn't in this gene.
          //console.log('even though connection exists globally, it isnt in this gene.');
          var c1 = R.zeros(3); // innovation number, weight, enabled (1)
          c1[IDX_CONNECTION] = searchIndex;
          c1[IDX_WEIGHT] = R.randn(initMu, initStdev);
          c1[IDX_ACTIVE] = 1;
          this.connections.push(c1);

          //console.log('added connection that exists somewhere else but not here.');
        }
      }

    },
    createUnrolledConnections: function() {
      // create a large array that is the size of Genome.connections
      // element:
      // 0: 1 or 0, whether this connection exists in this genome or not
      // 1: weight
      // 2: active? (1 or 0)
      var i, n, m, cIndex, c;
      this.unrolledConnections = [];
      n=connections.length; // global connection length
      m=this.connections.length;
      for (i=0;i<n;i++) {
        this.unrolledConnections.push(R.zeros(3));
      }
      for (i=0;i<m;i++) {
        c = this.connections[i];
        cIndex = c[IDX_CONNECTION];
        this.unrolledConnections[cIndex][IDX_CONNECTION] = 1;
        this.unrolledConnections[cIndex][IDX_WEIGHT] = c[IDX_WEIGHT];
        this.unrolledConnections[cIndex][IDX_ACTIVE] = c[IDX_ACTIVE];
      }
    },
    crossover: function(that) { // input is another genome
      // returns a newly create genome that is the offspring.
      var i, n, c;
      var child = new Genome();
      child.connections = []; // empty initial connections
      var g;
      var count;

      n = connections.length;

      this.createUnrolledConnections();
      that.createUnrolledConnections();

      for (i=0;i<n;i++) {
        count = 0;
        g = this;
        if (this.unrolledConnections[i][IDX_CONNECTION] === 1) {
          count++;
        }
        if (that.unrolledConnections[i][IDX_CONNECTION] === 1) {
          g = that;
          count++;
        }
        if (count === 2 && Math.random() < 0.5) {
          g = this;
        }
        if (count === 0) continue; // both genome doesn't contain this connection
        c = R.zeros(3);
        c[IDX_CONNECTION] = i;
        c[IDX_WEIGHT] = g.unrolledConnections[i][IDX_WEIGHT];
        // in the following line, the connection is disabled only of it is disabled on both parents
        c[IDX_ACTIVE] = 1;
        if (this.unrolledConnections[i][IDX_ACTIVE] === 0 && that.unrolledConnections[i][IDX_ACTIVE] === 0) {
          c[IDX_ACTIVE] = 0;
        }
        child.connections.push(c);
      }

      return child;
    },
    setupModel: function(inputDepth) {
      // setup recurrent.js model
      var i;
      var nNodes = nodes.length;
      var nConnections = connections.length;
      this.createUnrolledConnections();
      this.model = [];
      var nodeModel = [];
      var connectionModel = [];
      var c;
      for (i=0;i<nNodes;i++) {
        nodeModel.push(new R.Mat(inputDepth, 1));
      }
      for (i=0;i<nConnections;i++) {
        c = new R.Mat(1, 1);
        c.w[0] = this.unrolledConnections[i][IDX_WEIGHT];
        connectionModel.push(c);
      }
      this.model.nodes = nodeModel;
      this.model.connections = connectionModel;
    },
    updateModelWeights: function() {
      // assume setupModel is already run. updates internal weights
      // after backprop is performed
      var i, n, m, cIndex;
      var nConnections = connections.length;

      var connectionModel = this.model.connections;
      var c;

      for (i=0;i<nConnections;i++) {
        this.unrolledConnections[i][IDX_WEIGHT] = connectionModel[i].w[0];
      }

      m=this.connections.length;
      for (i=0;i<m;i++) {
        c = this.connections[i];
        cIndex = c[IDX_CONNECTION];
        if (c[IDX_ACTIVE]) {
          c[IDX_WEIGHT] = this.unrolledConnections[cIndex][IDX_WEIGHT];
        }
      }
    },
    zeroOutNodes: function() {
      R.zeroOutModel(this.model.nodes);
    },
    setInput: function(input) {
      // input is an n x d R.mat, where n is the inputDepth, and d is number of inputs
      // for generative art, d is typically just (x, y)
      // also sets all the biases to be 1.0
      // run this function _after_ setupModel() is called!
      var i, j;
      var n = input.n;
      var d = input.d;
      var inputNodeList = getNodeList(NODE_INPUT);
      var biasNodeList = getNodeList(NODE_BIAS);
      var dBias = biasNodeList.length;

      R.assert(inputNodeList.length === d, 'inputNodeList is not the same as dimentions');
      R.assert(this.model.nodes[0].n === n, 'model nodes is not the same as dimentions');

      for (i=0;i<n;i++) {
        for (j=0;j<d;j++) {
          this.model.nodes[inputNodeList[j]].set(i, 0, input.get(i, j));
        }
        for (j=0;j<dBias;j++) {
          this.model.nodes[biasNodeList[j]].set(i, 0, 1.0);
        }
      }
    },
    getOutput: function() {
      // returns an array of recurrent.js Mat's representing the output
      var i;
      var outputNodeList = getNodeList(NODE_OUTPUT);
      var d = outputNodeList.length;
      var output = [];
      for (i=0;i<d;i++) {
        output.push(this.model.nodes[outputNodeList[i]]);
      }
      return output;
    },
    roundWeights: function() {
      var precision = 10000;
      for (var i=0;i<this.connections.length;i++) {
        this.connections[i][IDX_WEIGHT] = Math.round(this.connections[i][IDX_WEIGHT]*precision)/precision;
      }
    },
    toJSON: function(description) {

      var data = {
        nodes: copyArray(nodes),
        connections: copyConnections(connections),
        nInput: nInput,
        nOutput: nOutput,
        renderMode: renderMode,
        outputIndex: outputIndex,
        genome: this.connections,
        description: description
      };

      this.backup = new Genome(this);

      return JSON.stringify(data);

    },
    fromJSON: function(data_string) {
      var data = JSON.parse(data_string);
      nodes = copyArray(data.nodes);
      connections = copyConnections(data.connections);
      nInput = data.nInput;
      nOutput = data.nOutput;
      renderMode = data.renderMode || 0; // might not exist.
      outputIndex = data.outputIndex;
      this.importConnections(data.genome);

      return data.description;
    },
    forward: function(G) {
      // forward props the network from input to output.  this is where magic happens.
      // input G is a recurrent.js graph
      var outputNodeList = getNodeList(NODE_OUTPUT);
      var biasNodeList = getNodeList(NODE_BIAS);
      var inputNodeList = biasNodeList.concat(getNodeList(NODE_INPUT));

      var i, j, n;
      var nNodes = nodes.length;
      var nConnections = connections.length;
      var touched = R.zeros(nNodes);
      var prevTouched = R.zeros(nNodes);
      var nodeConnections = new Array(nNodes); // array of array of connections.

      var nodeList = [];
      var binaryNodeList = R.zeros(nNodes);

      for (i=0;i<nNodes;i++) {
        nodeConnections[i] = []; // empty array.
      }

      for (i=0;i<nConnections;i++) {
        if (this.unrolledConnections[i][IDX_ACTIVE] && this.unrolledConnections[i][IDX_CONNECTION]) {
          nodeConnections[connections[i][1]].push(i); // push index of connection to output node
          binaryNodeList[connections[i][0]] = 1;
          binaryNodeList[connections[i][1]] = 1;
        }
      }

      for (i=0;i<nNodes;i++) {
        if (binaryNodeList[i] === 1) {
          nodeList.push(i);
        }
      }

      for (i=0,n=inputNodeList.length;i<n;i++) {
        touched[inputNodeList[i]] = 1.0;
      }

      function allTouched(listOfNodes) {
        for (var i=0,n=listOfNodes.length;i<n;i++) {
          if (touched[listOfNodes[i]] !== 1) {
            return false;
          }
        }
        return true;
      }

      function noProgress(listOfNodes) {
        var idx;
        for (var i=0,n=listOfNodes.length;i<n;i++) {
          idx = listOfNodes[i];
          if (touched[idx] !== prevTouched[idx]) {
            return false;
          }
        }
        return true;
      }

      function copyTouched(listOfNodes) {
        var idx;
        for (var i=0,n=listOfNodes.length;i<n;i++) {
          idx = listOfNodes[i];
          prevTouched[idx] = touched[idx];
        }
      }

      function forwardTouch() {
        var i, j;
        var n=nNodes, m, ix; // ix is the index of the global connections.
        var theNode;

        for (i=0;i<n;i++) {
          if (touched[i] === 0) {
            theNode = nodeConnections[i];
            for (j=0,m=theNode.length;j<m;j++) {
              ix = theNode[j];
              if (touched[connections[ix][0]] === 1) {
                //console.log('node '+connections[ix][0]+' is touched, so now node '+i+' has been touched');
                touched[i] = 2; // temp touch state
                break;
              }
            }
          }
        }

        for (i=0;i<n;i++) {
          if (touched[i] === 2) touched[i] = 1;
        }

      }

      // forward tick magic
      function forwardTick(model) {
        var i, j;
        var n, m, cIndex, nIndex; // ix is the index of the global connections.
        var theNode;

        var currNode, currOperand, currConnection; // recurrent js objects
        var needOperation; // don't need operation if node is operator(node) is null or mul or add
        var nodeType;
        var finOp; // operator after all operands are weighted summed or multiplied
        var op; // either 'add' or 'eltmult'
        var out; // temp variable for storing recurrentjs state
        var cumulate; // cumulate all the outs (either addition or mult)

        n=nNodes;
        for (i=0;i<n;i++) {
          if (touched[i] === 1) { // operate on this node since it has been touched

            theNode = nodeConnections[i];
            m=theNode.length;
            // if there are no operands for this node, then don't do anything.
            if (m === 0) continue;

            nodeType = nodes[i];
            needOperation = true;
            finOp = operators[nodeType];
            if (finOp === null || finOp === 'mult' || finOp === 'add' || nodeType === NODE_MGAUSSIAN) needOperation = false;

            // usually we add weighted sum of operands, except if operator is mult
            op = 'add';
            if (finOp === 'mult') op = 'eltmul';

            // cumulate all the operands
            for (j=0;j<m;j++) {
              cIndex = theNode[j];
              nIndex = connections[cIndex][0];
              currConnection = model.connections[cIndex];
              currOperand = model.nodes[nIndex];
              out = G.mul(currOperand, currConnection);
              if (nodeType === NODE_MGAUSSIAN) { // special case:  the nasty multi gaussian
                out = G.gaussian(out);
              }
              if (j === 0) { // assign first result to cumulate
                cumulate = out;
              } else { // cumulate next result after first operand
                cumulate = G[op](cumulate, out); // op is either add or eltmul
              }
            }

            // set the recurrentjs node here
            model.nodes[i] = cumulate;
            // operate on cumulated sum or product if needed
            if (needOperation) {
              model.nodes[i] = G[finOp](model.nodes[i]);
            }

            // another special case, squaring the output
            if (nodeType === NODE_SQUARE) {
              model.nodes[i] = G.eltmul(model.nodes[i], model.nodes[i]);
            }

          }
        }


      }

      function printTouched() {
        var i;
        var result="";
        for (i=0;i<touched.length;i++) {
          result += touched[i]+" ";
        }
        console.log(result);
      }

      //printTouched();
      for (i=0;i<MAX_TICK;i++) {
        forwardTouch();
        forwardTick(this.model); // forward tick the network using graph
        //printTouched();
        /*
        if (allTouched(outputNodeList)) {
          //console.log('all outputs touched!');
          //break;
        }
        */
        if (allTouched(nodeList)) {
          //console.log('all nodes touched!');
          break;
        }
        if (noProgress(nodeList)) { // the forward tick made no difference, stuck
          //console.log('all nodes touched!');
          break;
        }
        copyTouched(nodeList);
      }

    }

  };

  var NEATCompressor = function() {
    // compresses neat, given a list of genes.
  };

  NEATCompressor.prototype = {
    buildMap: function(genes) {
      // pass in an array of all the genomes that matter, to build compression map.
      var nNode = nodes.length;
      var nConnection = connections.length;
      var nGene = genes.length;
      var connectionUsage = R.zeros(nConnection);
      var nodeUsage = R.zeros(nNode);
      var i, j, gc, c, idx, nodeIndex1, nodeIndex2;
      var newConnectionCount = 0;
      var newNodeCount = 0;
      // find out which connections are actualy used by the population of genes
      for (i=0;i<nGene;i++) {
        gc = genes[i].connections;
        for (j=0;j<gc.length;j++) {
          if (gc[j][IDX_ACTIVE] === 1) {
            idx = gc[j][IDX_CONNECTION]; // index of global connections array.
            connectionUsage[idx] = 1;
          }
        }
      }
      // from the active connections, find out which nodes are actually used
      for (i=0;i<nConnection;i++) {
        if (connectionUsage[i] === 1) {
          newConnectionCount += 1;
          nodeIndex1 = connections[i][0]; // from node
          nodeIndex2 = connections[i][1]; // to node
          nodeUsage[nodeIndex1] = 1;
          nodeUsage[nodeIndex2] = 1;
        }
      }
      // count active nodes
      for (i=0;i<nNode;i++) {
        if (nodeUsage[i] === 1) {
          newNodeCount += 1;
        }
      }
      // declare maps
      this.nodeMap = R.zeros(newNodeCount);
      this.connectionMap = R.zeros(newConnectionCount);
      this.nodeReverseMap = R.zeros(nNode);
      this.connectionReverseMap = R.zeros(nConnection);
      // calculate maps
      j = 0;
      for (i=0;i<nNode;i++) {
        if (nodeUsage[i] === 1) {
          this.nodeMap[j] = i;
          this.nodeReverseMap[i] = j;
          j += 1;
        }
      }
      j = 0;
      for (i=0;i<nConnection;i++) {
        if (connectionUsage[i] === 1) {
          this.connectionMap[j] = i;
          this.connectionReverseMap[i] = j;
          j += 1;
        }
      }

      // calculate new nodes and connections arrays but store them in compressor
      // only replace live ones when comressNEAT() is called.
      this.newNodes = [];
      this.newConnections = [];
      for (i=0;i<newNodeCount;i++) {
        this.newNodes.push(nodes[this.nodeMap[i]]);
      }
      for (i=0;i<newConnectionCount;i++) {
        c = connections[this.connectionMap[i]];
        nodeIndex1 = this.nodeReverseMap[c[0]]; // fix bug here.
        nodeIndex2 = this.nodeReverseMap[c[1]];
        this.newConnections.push([nodeIndex1, nodeIndex2]);
      }
    },
    compressNEAT: function() {
      // compresses nodes and connections global vars in neat.js
      /* ie, these:
      var nodes = []; // this array holds all nodes
      var connections = []; // index of connections here is the 'innovation' value
      */
      nodes = this.newNodes;
      connections = this.newConnections;
    },
    compressGenes: function(genes) {
      // applies the compression map to an array of genomes
      var nGene = genes.length;
      var newConnections = [];
      var gc, c, oldc, i, j, w;
      var oldConnectionIndex;

      for (i=0;i<nGene;i++) {
        gc = genes[i].connections;
        newConnections = [];
        for (j=0;j<gc.length;j++) {
          oldc = gc[j];
          if (oldc[IDX_ACTIVE] === 1) {
            oldConnectionIndex = oldc[IDX_CONNECTION];
            w = oldc[IDX_WEIGHT];
            c = R.zeros(3); // innovation number, weight, enabled (1)
            c[IDX_CONNECTION] = this.connectionReverseMap[oldConnectionIndex];
            c[IDX_WEIGHT] = w;
            c[IDX_ACTIVE] = 1;
            newConnections.push(c);
          }
        }
        genes[i].connections = newConnections;
      }
    },
  };

  var NEATTrainer = function(options_, initGenome_) {
    // implementation of a variation of NEAT training algorithm based off K-medoids.
    //
    // options:
    // num_populations : positive integer, the number of sub populations we want to preserve.
    // sub_population_size : positive integer.  Note that this is the population size for each sub population
    // hall_of_fame_size : positive integer, stores best guys in all of history and keeps them.
    // new_node_rate : [0, 1], when mutation happens, chance of a new node being added
    // new_connection_rate : [0, 1], when mutation happens, chance of a new connection being added
    // extinction_rate : [0, 1], probability that crappiest subpopulation is killed off during evolve()
    // mutation_rate : [0, 1], when mutation happens, chance of each connection weight getting mutated
    // mutation_size : positive floating point.  stdev of gausian noise added for mutations
    // init_weight_magnitude : stdev of initial random weight (default = 1.0)
    // debug_mode: false by default.  if set to true, console.log debug output occurs.
    // target_fitness : after fitness achieved is greater than this float value, learning stops
    // initGenome_: model NEAT genome to initialize with. can be result obtained from pretrained sessions.

    var options = options_ || {};

    this.num_populations = typeof options.num_populations !== 'undefined' ? options.num_populations : 5;
    this.sub_population_size = typeof options.sub_population_size !== 'undefined' ? options.sub_population_size : 10;
    this.hall_of_fame_size = typeof options.hall_of_fame_size !== 'undefined' ? options.hall_of_fame_size : 5;

    this.new_node_rate = typeof options.new_node_rate !== 'undefined' ? options.new_node_rate : 0.1;
    this.new_connection_rate = typeof options.new_connection_rate !== 'undefined' ? options.new_connection_rate : 0.1;
    this.extinction_rate = typeof options.extinction_rate !== 'undefined' ? options.extinction_rate : 0.5;
    this.mutation_rate = typeof options.mutation_rate !== 'undefined' ? options.mutation_rate : 0.1;
    this.mutation_size = typeof options.mutation_size !== 'undefined' ? options.mutation_size : 1.0;
    this.init_weight_magnitude = typeof options.init_weight_magnitude !== 'undefined' ? options.init_weight_magnitude : 1.0;

    this.target_fitness = typeof options.target_fitness !== 'undefined' ? options.target_fitness : 1e20;

    this.debug_mode = typeof options.debug_mode !== 'undefined' ? options.debug_mode : false;

    // module globals should be changed as well
    initMu = 0.0;
    initStdev = this.init_weight_magnitude; // randomised param initialisation.
    mutationRate = this.mutation_rate;
    mutationSize = this.mutation_size;

    // if the below is set to true, then extinction will be turned on for the next evolve()
    this.forceExtinctionMode = false;

    var genome;
    var i, N, K;

    N = this.sub_population_size;
    K = this.num_populations;

    kmedoids.init(K);
    kmedoids.setDistFunction(this.dist);

    this.genes = []; // population
    this.hallOfFame = []; // stores the hall of fame here!
    this.bestOfSubPopulation = []; // stores the best gene for each sub population here.

    this.compressor = new NEATCompressor(); // this guy helps compress the network.

    // generates the initial genomes
    for (i = 0; i < N*K; i++) {

      if (typeof initGenome_ !== 'undefined') {
        genome = new Genome(initGenome_);
      } else {
        genome = new Genome(); // empty one with no connections
      }

      // initially, just create a single connection from input or bias to outputs
      genome.addRandomConnection();
      genome.mutateWeights(1.0, this.mutation_size); // burst mutate init weights

      // stamp meta info into the genome
      genome.fitness = -1e20;
      genome.cluster = R.randi(0, K);
      this.genes.push(genome);

    }

    for (i = 0; i < this.hall_of_fame_size; i++) {
      if (typeof initGenome_ !== 'undefined') {
        genome = new Genome(initGenome_); // don't modify old results in hof
      } else {
        genome = new Genome(); // empty one with no connections

        // initially, just create a single connection from input or bias to outputs
        genome.addRandomConnection();
        genome.mutateWeights(1.0, this.mutation_size); // burst mutate init weights
      }

      // stamp meta info into the genome
      genome.fitness = -1e20;
      genome.cluster = 0; //R.randi(0, K);
      this.hallOfFame.push(genome);
    }


  };

  NEATTrainer.prototype = {
    sortByFitness: function(c) {
      c = c.sort(function (a, b) {
        if (a.fitness > b.fitness) { return -1; }
        if (a.fitness < b.fitness) { return 1; }
        return 0;
      });
    },
    forceExtinction: function() {
      this.forceExtinctionMode = true;
    },
    resetForceExtinction: function() {
      this.forceExtinctionMode = false;
    },
    applyMutations: function(g) {
      // apply mutations (new node, new connection, mutate weights) on a specified genome g
      if (Math.random() < this.new_node_rate) g.addRandomNode();
      if (Math.random() < this.new_connection_rate) g.addRandomConnection();
      g.mutateWeights(this.mutation_rate, this.mutation_size);
    },
    applyFitnessFuncToList: function(f, geneList) {
      var i, n;
      var g;
      for (i=0,n=geneList.length;i<n;i++) {
        g = geneList[i];
        g.fitness = f(g);
      }
    },
    getAllGenes: function() {
      // returns the list of all the genes plus hall(s) of fame
      return this.genes.concat(this.hallOfFame).concat(this.bestOfSubPopulation);
    },
    applyFitnessFunc: function(f, _clusterMode) {
      // applies fitness function f on everyone including hall of famers
      // in the future, have the option to avoid hall of famers
      var i, n;
      var j, m;
      var g;
      var K = this.num_populations;

      var clusterMode = true; // by default, we would cluster stuff (takes time)
      if (typeof _clusterMode !== 'undefined') {
        clusterMode = _clusterMode;
      }

      this.applyFitnessFuncToList(f, this.genes);
      this.applyFitnessFuncToList(f, this.hallOfFame);
      this.applyFitnessFuncToList(f, this.bestOfSubPopulation);

      this.filterFitness();
      this.genes = this.genes.concat(this.hallOfFame);
      this.genes = this.genes.concat(this.bestOfSubPopulation);
      this.sortByFitness(this.genes);

      // cluster before spinning off hall of fame:
      if (clusterMode) {
        this.cluster();
      }

      // rejig hall of fame
      this.hallOfFame = [];
      for (i=0,n=this.hall_of_fame_size;i<n;i++) {
        g = this.genes[i].copy();
        g.fitness = this.genes[i].fitness;
        g.cluster = this.genes[i].cluster;
        this.hallOfFame.push(g);
      }

      // store best of each sub population (may be overlaps with hall of fame)
      this.bestOfSubPopulation = [];
      for (j=0;j<K;j++) {
        for (i=0,n=this.genes.length;i<n;i++) {
          if (this.genes[i].cluster === j) {
            g = this.genes[i].copy();
            g.fitness = this.genes[i].fitness;
            g.cluster = this.genes[i].cluster;
            this.bestOfSubPopulation.push(g);
            break;
          }
        }
      }

    },
    clipWeights: function(maxWeight_) {
      // applies fitness function f on everyone including hall of famers
      // in the future, have the option to avoid hall of famers
      var i, n;
      var g;
      for (i=0,n=this.genes.length;i<n;i++) {
        g = this.genes[i];
        g.clipWeights(maxWeight_);
      }
      for (i=0,n=this.hallOfFame.length;i<n;i++) {
        g = this.hallOfFame[i];
        g.clipWeights(maxWeight_);
      }
    },
    areWeightsNaN: function() {
      // if any weight value is NaN of any gene, then returns true and break.
      var i, n;
      var g;
      for (i=0,n=this.genes.length;i<n;i++) {
        g = this.genes[i];
        if (g.areWeightsNaN()) return true;
      }
      for (i=0,n=this.hallOfFame.length;i<n;i++) {
        g = this.hallOfFame[i];
        if (g.areWeightsNaN()) return true;
      }
      return false;
    },
    filterFitness: function() {
      // achieves 2 things. converts NaN to -1e20
      // makes sure all fitness numbers have negative values
      // makes sure each fitness number is less than minus epsilon
      // the last point is important, since in NEAT, we will randomly choose
      // parents based on their inverse fitness normalised probabilities

      var i, n;
      var epsilon = 1e-10;
      var g;
      function tempProcess(g) {
        var fitness = -1e20;
        if (typeof g.fitness !== 'undefined' && isNaN(g.fitness) === false) {
          fitness = -Math.abs(g.fitness);
          fitness = Math.min(fitness, -epsilon);
        }
        g.fitness = fitness;
      }
      for (i=0,n=this.genes.length;i<n;i++) {
        g = this.genes[i];
        tempProcess(g);
      }
      for (i=0,n=this.hallOfFame.length;i<n;i++) {
        g = this.hallOfFame[i];
        tempProcess(g);
      }
    },
    pickRandomIndex: function(genes, cluster_) {
      // Ehis function returns a random index for a given gene array 'genes'
      // Each element of genes will have a strictly negative .fitness parameter
      // the picking will be probability weighted to the fitness
      // A .normFitness parameter will be tagged onto each element
      // If cluster_ is specified, then each element of genes will be assumed
      // to have a .cluster parameter, and the resulting index will be from
      // that cluster.  if not, then all elements will be elegible
      // Assumes that filterFitness is run to clean the .fitness values up.
      var i, n;
      var byCluster = false;
      var cluster = 0;
      var totalProb = 0;
      var g;

      if (typeof cluster_ !== 'undefined') {
        byCluster = true;
        cluster = cluster_;
      }
      n = genes.length;

      var slack = 0.01; // if this is set to > 0, it ensure that the very best solution won't be picked each time.
      // if it is 0, and the best solution has a solution closer t -0, then best solution will be picked each time.

      // create inverse fitnesses
      for (i=0;i<n;i++) {
        g = genes[i];
        if (byCluster === false || g.cluster === cluster) {
          g.normFitness = 1/(-g.fitness+slack);
          //g.normFitness *= g.normFitness; // square this bitch, so crappy solutions have less chances...
          totalProb += g.normFitness;
        }
      }

      // normalize each fitness
      for (i=0;i<n;i++) {
        g = genes[i];
        if (byCluster === false || g.cluster === cluster) {
          g.normFitness /= totalProb;
        }
      }

      var x = Math.random(); // x will be [0, 1)
      var idx = -1;

      // find the index that corresponds to probability x
      for (i=0;i<n;i++) {
        g = genes[i];
        if (byCluster === false || g.cluster === cluster) {
          x -= g.normFitness;
          if (x <= 0) {
            idx = i;
            break;
          }
        }
      }

      return idx;
    },
    cluster: function(genePool_) {
      // run K-medoids algorithm to cluster this.genes
      var genePool = this.genes;
      var i, j, m, n, K, idx;
      if (typeof genePool_ !== 'undefined') genePool = genePool_;
      kmedoids.partition(this.genes); // split into K populations

      K = this.num_populations;

      var clusterIndices = kmedoids.getCluster();

      // put everything into new gene clusters
      for (i=0;i<K;i++) {
        m = clusterIndices[i].length;
        for (j=0;j<m;j++) {
          idx = clusterIndices[i][j];
          genePool[idx].cluster = i;
        }
      }
    },
    evolve: function(_mutateWeightsOnly) {
      // this is where the magic happens!
      //
      // performs one step evolution of the entire population
      //
      // assumes that applyFitnessFunc() or .fitness vals have been populated
      // .fitness values must be strictly negative.

      // concats both genes and hallOfFame into a combined bigger genepool
      //var prevGenes = this.genes.concat(this.hallOfFame);

      // assumes that clustering is already done!  important.
      // so assumes that the .cluster value for each genome is assigned.
      var prevGenes = this.genes;
      var newGenes = []; // new population array
      var i, n, j, m, K, N, idx;

      var worstFitness = 1e20;
      var worstCluster = -1;

      var bestFitness = -1e20;
      var bestCluster = -1;

      var mutateWeightsOnly = false;

      if (typeof _mutateWeightsOnly !== 'undefined') {
        mutateWeightsOnly = _mutateWeightsOnly;
      }

      K = this.num_populations;
      N = this.sub_population_size;

      // increase the generaiton:
      incrementGenerationCounter();

      var clusterIndices = kmedoids.getCluster();

      var cluster = new Array(K);

      // put everything into new gene clusters
      for (i=0;i<K;i++) {
        m = clusterIndices[i].length;
        cluster[i] = new Array(m);
        for (j=0;j<m;j++) {
          idx = clusterIndices[i][j];
          cluster[i][j] = prevGenes[idx];
        }
        this.sortByFitness(cluster[i]);

        // determine worst cluster (to destroy that sub population)
        if (cluster[i][0].fitness < worstFitness) {
          worstFitness = cluster[i][0].fitness;
          worstCluster = i;
        }

        // determine best cluster
        if (cluster[i][0].fitness >= bestFitness) {
          bestFitness = cluster[i][0].fitness;
          bestCluster = i;
        }
      }

      var mom, dad, baby, momIdx, dadIdx;

      // whether to kill off crappiest sub population and replace with best sub population
      // if we are just evolving weights only (CNE) then no need for extinction.
      var extinctionEvent = false;
      if (Math.random() < this.extinction_rate && mutateWeightsOnly === false) {
        extinctionEvent = true;
        if (this.debug_mode) console.log('the crappiest sub population will be made extinct now!');
      }
      if (this.forceExtinctionMode && mutateWeightsOnly === false) {
        if (this.debug_mode) console.log('forced extinction of crappiest sub population.');
        extinctionEvent = true;
      }

      for (i=0;i<K;i++) {

        // go thru each cluster, and mate N times with 2 random parents each time
        // if it is the worst cluster, then use everything.
        for (j=0;j<N;j++) {
          if (extinctionEvent && i === worstCluster) {
            momIdx = this.pickRandomIndex(prevGenes,bestCluster);
            dadIdx = this.pickRandomIndex(prevGenes,bestCluster);
          } else {
            momIdx = this.pickRandomIndex(prevGenes,i);
            dadIdx = this.pickRandomIndex(prevGenes,i);
          }
          mom = prevGenes[momIdx];
          dad = prevGenes[dadIdx];

          try {

          if (mutateWeightsOnly) {
            baby = mom.crossover(dad);
            //baby = mom.copy();
            baby.mutateWeights(this.mutation_rate, this.mutation_size);
          } else {
            baby = mom.crossover(dad);
            this.applyMutations(baby);
          }

          } catch (err) {
            if (this.debug_mode) {
              console.log("Error with mating: "+err);
              console.log("momIdx = "+momIdx);
              console.log("dadIdx = "+dadIdx);
              console.log("mom:");
              console.log(mom);
              console.log("dad:");
              console.log(dad);
            }
            baby = this.getBestGenome(i).copy();
            this.applyMutations(baby);
          }
          finally {
            baby.cluster = i;
            newGenes.push(baby);
          }
        }
      }

      this.genes = newGenes;

      this.compressor.buildMap(this.getAllGenes());
      this.compressor.compressNEAT();
      this.compressor.compressGenes(this.genes);
      this.compressor.compressGenes(this.hallOfFame);
      this.compressor.compressGenes(this.bestOfSubPopulation);

    },
    printFitness: function() {
      // debug function to print out fitness for all genes and hall of famers
      var i, n;
      var g;
      for (i=0,n=this.genes.length;i<n;i++) {
        g = this.genes[i];
        console.log('genome '+i+' fitness = '+g.fitness);
      }
      for (i=0,n=this.hallOfFame.length;i<n;i++) {
        g = this.hallOfFame[i];
        console.log('hallOfFamer '+i+' fitness = '+g.fitness);
      }
      for (i=0,n=this.bestOfSubPopulation.length;i<n;i++) {
        g = this.bestOfSubPopulation[i];
        console.log('bestOfSubPopulation '+i+' fitness = '+g.fitness);
      }
    },
    getBestGenome: function(cluster_) {
      // returns the b
      var bestN = 0;
      var cluster = 0;
      var i, n;
      var g;

      var allGenes = this.genes;
      this.sortByFitness(allGenes);
      if (typeof cluster_ === 'undefined') {
        return allGenes[bestN];
      }
      cluster = cluster_;
      for (i=0,n=allGenes.length;i<n;i++) {
        g = allGenes[i];
        if (g.cluster === cluster) {
          bestN = i;
          break;
        }
      }
      return allGenes[bestN];
    },
    dist: function(g1, g2) { // calculates 'distance' between 2 genomes
      g1.createUnrolledConnections();
      g2.createUnrolledConnections();

      var coef = { // coefficients for determining distance
        excess : 10.0,
        disjoint : 10.0,
        weight : 0.1,
      };

      var i, n, c1, c2, exist1, exist2, w1, w2, lastIndex1, lastIndex2, minIndex, active1, active2;
      //var active1, active2;
      var nBothActive = 0;
      var nDisjoint = 0;
      var nExcess = 0;
      var weightDiff = 0;
      var unrolledConnections1 = [];
      var unrolledConnections2 = [];
      n=connections.length; // global connection length

      var diffVector = R.zeros(n);

      for (i=0;i<n;i++) {
        c1 = g1.unrolledConnections[i];
        c2 = g2.unrolledConnections[i];
        exist1 = c1[IDX_CONNECTION];
        exist2 = c2[IDX_CONNECTION];
        active1 = exist1*c1[IDX_ACTIVE];
        active2 = exist2*c2[IDX_ACTIVE];
        if (exist1 === 1) lastIndex1 = i;
        if (exist2 === 1) lastIndex2 = i;
        diffVector[i] = (exist1 === exist2)? 0 : 1; // record if one is active and the other is not
        if (active1 === 1 && active2 === 1) { // both active (changed to exist)
          w1 = c1[IDX_WEIGHT];
          w2 = c2[IDX_WEIGHT];
          R.assert(!isNaN(w1), 'weight1 inside dist function is NaN.');
          R.assert(!isNaN(w2), 'weight2 inside dist function is NaN.');
          nBothActive += 1;
          weightDiff += Math.abs(w1 - w2);
        }
      }
      minIndex = Math.min(lastIndex1, lastIndex2);
      if (nBothActive > 0) weightDiff /= nBothActive; // calculate average weight diff

      for (i=0;i<=minIndex;i++) {
        // count disjoints
        nDisjoint += diffVector[i];
      }
      for (i=minIndex+1;i<n;i++) {
        // count excess
        nExcess += diffVector[i];
      }

      var numNodes = Math.max(g1.getNodesInUse().length,g2.getNodesInUse().length);
      var distDisjoint = coef.disjoint*nDisjoint / numNodes;
      var distExcess = coef.excess*nExcess / numNodes;
      var distWeight = coef.weight * weightDiff;
      var distance = distDisjoint+distExcess+distWeight;

      if (isNaN(distance) || Math.abs(distance) > 100) {
        console.log('large distance report:');
        console.log('distance = '+distance);
        console.log('disjoint = '+distDisjoint);
        console.log('excess = '+distExcess);
        console.log('weight = '+distWeight);
        console.log('numNodes = '+numNodes);
        console.log('nBothActive = '+nBothActive);
      }

      /*
      console.log('distance calculation');
      console.log('nDisjoint = '+nDisjoint);
      console.log('nExcess = '+nExcess);
      console.log('avgWeightDiff = '+weightDiff);
      console.log('distance = '+distance);
      */

      return distance;
    },
  };

  global.init = init;
  global.Genome = Genome;
  global.getNodes = getNodes;
  global.getConnections = getConnections;
  global.randomizeRenderMode = randomizeRenderMode;
  global.setRenderMode = setRenderMode;
  global.getRenderMode = getRenderMode;
  global.NEATTrainer = NEATTrainer;
  global.NEATCompressor = NEATCompressor;
  global.getNumInput = function() { return nInput; };
  global.getNumOutput = function() { return nOutput; };
  global.incrementGenerationCounter = incrementGenerationCounter;
  global.getNumGeneration = function() { return generationNum; };


})(N);
(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    //window.jsfeat = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(N);


