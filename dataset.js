// generates random dummy dataset, and provides an interface to easily access that data from example.js
// data generation based on https://github.com/tensorflow/playground
// adopted from https://github.com/tensorflow/playground/blob/master/dataset.ts

if (typeof module != "undefined") {
  var R = require('./ml/recurrent.js');
  var kmedoids = require('./ml/kmedoids.js');
  var N = require('./ml/neat.js');
}

var DataSet = {};

(function(global) {
  "use strict";

  // settings
  var nSize = 200; // training size
  var nTestSize = 200; // testing size
  var noiseLevel = 0.5;

  // minibatch for training, randomly population from trainList.
  var nBatch = 10; // minibatch size
  var dataBatch = new R.Mat(nBatch, 2);
  var labelBatch = new R.Mat(nBatch, 1);

  // dataset is stored here
  var trainList = [];
  var testList = [];

  // R.Mat format of train and test data
  var trainData;
  var trainLabel;
  var testData;
  var testLabel;

  var DataPoint = function(x, y, l) {
    this.x = x;
    this.y = y;
    this.l = l; // label
  };

  var shuffleDataList = function(pList) {
    var i, randomIndex;
    var p, t, px, py, pl;
    var N = pList.length;
    for (i=0;i<N;i++) {
      // swaps i with randomIndex
      randomIndex = R.randi(0, N);
      p = pList[randomIndex];
      t = pList[i];
      px = p.x; py = p.y; pl = p.l;
      p.x = t.x; p.y = t.y; p.l = t.l;
      t.x = px; t.y = py; t.l = pl;
    }
  };

  function generateXORData(numPoints_, noise_) {
    var particleList = [];
    var i, N;
    var x, y, l;
    N = typeof numPoints_ === 'undefined' ? nSize : numPoints_;
    var noise = typeof noise_ === 'undefined' ? 0.5 : noise_;
    for (i = 0; i < N; i++) {
      x = R.randf(-5.0, 5.0)+R.randn(0, noise);
      y = R.randf(-5.0, 5.0)+R.randn(0, noise);
      l = 0;
      if (x > 0 && y > 0) l = 1;
      if (x < 0 && y < 0) l = 1;
      particleList.push(new DataPoint(x, y, l));
    }
    return particleList;
  }

  function generateSpiralData(numPoints_, noise_) {
    var particleList = [];
    var noise = typeof noise_ === 'undefined' ? 0.5 : noise_;
    var N = typeof numPoints_ === 'undefined' ? nSize : numPoints_;

    function genSpiral(deltaT, l) {
      var n = N / 2;
      var r, t, x, y;
      for (var i = 0; i < n; i++) {
        r = i / n * 6.0;
        t = 1.75 * i / n * 2 * Math.PI + deltaT;
        x = r * Math.sin(t) + R.randf(-1, 1) * noise;
        y = r * Math.cos(t) + R.randf(-1, 1) * noise;
        particleList.push(new DataPoint(x, y, l));
      }
    }

    var flip = 0; // R.randi(0, 2);
    var backside = 1-flip;
    genSpiral(0, flip); // Positive examples.
    genSpiral(Math.PI, backside); // Negative examples.
    return particleList;
  }

  function generateGaussianData(numPoints_, noise_) {
    var particleList = [];
    var noise = typeof noise_ === 'undefined' ? 0.5 : noise_;
    var N = typeof numPoints_ === 'undefined' ? nSize : numPoints_;

    function genGaussian(xc, yc, l) {
      var n = N / 2;
      var x, y;
      for (var i = 0; i < n; i++) {
        x = R.randn(xc, noise*1.0+1.0);
        y = R.randn(yc, noise*1.0+1.0);
        particleList.push(new DataPoint(x, y, l));
      }
    }
    genGaussian(2*1, 2*1, 1); // Positive examples.
    genGaussian(-2*1, -2*1, 0); // Negative examples.
    return particleList;
  }
  function generateCircleData(numPoints_, noise_) {
    var particleList = [];
    var noise = typeof noise_ === 'undefined' ? 0.5 : noise_;
    var N = typeof numPoints_ === 'undefined' ? nSize : numPoints_;
    var n = N / 2;
    var i, r, x, y, l, angle, noiseX, noiseY;
    var radius = 5.0;

    function getCircleLabel(x, y) {
      return (x*x+y*y < (radius * 0.5)*(radius * 0.5)) ? 1 : 0;
    }

    // Generate positive points inside the circle.
    for (i = 0; i < n; i++) {
      r = R.randf(0, radius * 0.5);
      angle = R.randf(0, 2 * Math.PI);
      x = r * Math.sin(angle);
      y = r * Math.cos(angle);
      noiseX = R.randf(-radius, radius) * noise/3;
      noiseY = R.randf(-radius, radius) * noise/3;
      l = getCircleLabel(x, y);
      particleList.push(new DataPoint(x+noiseX, y+noiseY, l));
    }

    // Generate negative points outside the circle.
    for (i = 0; i < n; i++) {
      r = R.randf(radius * 0.75, radius);
      angle = R.randf(0, 2 * Math.PI);
      x = r * Math.sin(angle);
      y = r * Math.cos(angle);
      noiseX = R.randf(-radius, radius) * noise/3;
      noiseY = R.randf(-radius, radius) * noise/3;
      l = getCircleLabel(x, y);
      particleList.push(new DataPoint(x+noiseX, y+noiseY, l));
    }
    return particleList;
  }

  var convertData = function() {
    // converts the data to R.Mat format
    var p; // datapoint
    var i; // counter
    testData = new R.Mat(nTestSize, 2);
    testLabel = new R.Mat(nTestSize, 1);
    trainData = new R.Mat(nSize, 2);
    trainLabel = new R.Mat(nSize, 1);

    for (i=0;i<nTestSize;i++) {
      p = testList[i];
      testData.set(i, 0, p.x);
      testData.set(i, 1, p.y);
      testLabel.w[i] = p.l;
    }
    for (i=0;i<nSize;i++) {
      p = trainList[i];
      trainData.set(i, 0, p.x);
      trainData.set(i, 1, p.y);
      trainLabel.w[i] = p.l;
    }
  };

  var generateMiniBatch = function() {
    // generates training minibatch
    var i, N=nSize, p, randomIndex;

    for (i=0;i<nBatch;i++) {
      randomIndex = R.randi(0, N);
      p = trainList[randomIndex];
      dataBatch.set(i, 0, p.x);
      dataBatch.set(i, 1, p.y);
      labelBatch.w[i] = p.l;
    }
  };

  var generateRandomData = function(choice_) {
    var choice = R.randi(0, 4);
    if (typeof choice_ != "undefined") choice = choice_;
    if (choice === 0) {
      trainList = generateCircleData(nSize, noiseLevel);
      testList = generateCircleData(nTestSize, noiseLevel);
    } else if (choice === 1) {
      trainList = generateXORData(nSize, noiseLevel);
      testList = generateXORData(nTestSize, noiseLevel);
    } else if (choice === 2) {
      trainList = generateGaussianData(nSize, noiseLevel);
      testList = generateGaussianData(nTestSize, noiseLevel);
    } else {
      trainList = generateSpiralData(nSize, noiseLevel);
      testList = generateSpiralData(nTestSize, noiseLevel);
    }
    shuffleDataList(trainList);
    shuffleDataList(testList);
    convertData();
  };

  global.generateRandomData = generateRandomData;
  global.generateMiniBatch = generateMiniBatch;
  global.getTrainData = function() { return trainData };
  global.getTrainLabel = function() { return trainLabel };
  global.getTestData = function() { return testData };
  global.getTestLabel = function() { return testLabel };
  global.getBatchData = function() { return dataBatch };
  global.getBatchLabel = function() { return labelBatch };
  global.getTrainLength = function() { return nSize };
  global.getTestLength = function() { return nTestSize };
  global.getBatchLength = function() { return nBatch };

})(DataSet);
(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    //window.jsfeat = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(DataSet);
