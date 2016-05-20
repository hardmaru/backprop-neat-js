
var md, desktopMode = true;

var dataFactor = 24;
var dataWidth = 320;

function p2d(x) { // pixel to data
  return (x - dataWidth/2) / dataFactor;
}

function d2p(x) { // data to pixel
  return x * dataFactor + dataWidth/2;
}

var nBackprop = 600;
var learnRate = 0.01;

var trainer;

var genome, input;
var colaGraph;
var modelReady = false;
var selectedCluster = -1;

// fitness penalties
var penaltyNodeFactor = 0.00;
var penaltyConnectionFactor = 0.03;
var noiseLevel = 0.5;
var makeConnectionProbability = 0.5;
var makeNodeProbability = 0.2;

// presentation mode
var presentationMode = false;

// do a parameter update on W,b:
var solver = new R.Solver(); // the Solver uses RMSProp

// update W and b, use learning rate of 0.01,
// regularization strength of 0.0001 and clip gradient magnitudes at 5.0

md = new MobileDetect(window.navigator.userAgent);
if (md.mobile()) {
    desktopMode = false;
    console.log('mobile: '+md.mobile());
    dataWidth=160;
    dataFactor=12;
    $("#warningText").show();
} else {
    desktopMode = true;
    console.log('not mobile');
}

var Particle = function(x, y, l) {
  this.x = x;
  this.y = y;
  this.l = l;
};

// data settings
var initNSize = 200;
var nSize = initNSize; // training size
var particleList = [];
var predictionList;
var accuracy = 0.0;
var data; // holds particleList's x training data
var label; // holds particleList's y training data
var showTrainData = true;

var dataSetChoice = 0;

// test set
var initNTestSize = 400;
var nTestSize = initNTestSize; // test size
var particleTestList = [];
var predictionTestList;
var testAccuracy = 0.0;
var testData; // holds particleList's x training data
var testLabel; // holds particleList's y training data
var showTestData = true;

// minibatch
var nBatch = 10; // minibatch size
var dataBatch = new R.Mat(nBatch, 2);
var labelBatch = new R.Mat(nBatch, 1);

// custom data mode
var requiredCustomData = 40;

// prediction image
var img;
var imgData;
var imgPrediction;
function createImgData() {
  var x, y;
  var N = (dataWidth/2-1);
  imgData = new R.Mat(N*N, 2);
  for (y=0;y<N;y++) {
    for (x=0;x<N;x++) {
      imgData.set(y*N+x, 0, p2d(2*x));
      imgData.set(y*N+x, 1, p2d(2*y));
    }
  }
  imgPrediction = new R.zeros(N*N);
}
createImgData();

var shuffleParticleList = function(pList) {
  // shuffle particleList. (sorry global, crappy style, so kill me..)
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
var makeDataLabel = function() {
  // dumps particleList into data and label array for easy processing.
  // put particleList in optimised R.Mat formats below.
  var i, n=particleList.length, p;
  predictionList = R.zeros(n)
  data = new R.Mat(n, 2);
  label = new R.Mat(n, 1);

  for (i=0;i<n;i++) {
    p = particleList[i];
    data.set(i, 0, p.x);
    data.set(i, 1, p.y);
    label.w[i] = p.l;
  }
  // sloppy code below.
  n=particleTestList.length;
  predictionTestList = R.zeros(n)
  testData = new R.Mat(n, 2);
  testLabel = new R.Mat(n, 1);

  for (i=0;i<n;i++) {
    p = particleTestList[i];
    testData.set(i, 0, p.x);
    testData.set(i, 1, p.y);
    testLabel.w[i] = p.l;
  }
};
var makeMiniBatch = function() {
  // generates training minibatch
  var i, N=particleList.length, p, randomIndex;

  for (i=0;i<nBatch;i++) {
    randomIndex = R.randi(0, N);
    p = particleList[randomIndex];
    dataBatch.set(i, 0, p.x);
    dataBatch.set(i, 1, p.y);
    labelBatch.w[i] = p.l;
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
    particleList.push(new Particle(x, y, l));
  }
  return particleList;
}

// adopted from https://github.com/tensorflow/playground/blob/master/dataset.ts
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
      particleList.push(new Particle(x, y, l));
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
      particleList.push(new Particle(x, y, l));
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
    particleList.push(new Particle(x+noiseX, y+noiseY, l));
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
    particleList.push(new Particle(x+noiseX, y+noiseY, l));
  }
  return particleList;
}

function generateRandomData() {
  nTestSize = initNTestSize;
  nSize = initNSize;
  var choice = dataSetChoice; //R.randi(0, 4);
  if (choice === 0) {
    particleList = generateCircleData(nSize, noiseLevel);
    particleTestList = generateCircleData(nTestSize, noiseLevel);
  } else if (choice === 1) {
    particleList = generateXORData(nSize, noiseLevel);
    particleTestList = generateXORData(nTestSize, noiseLevel);
  } else if (choice === 2) {
    particleList = generateGaussianData(nSize, noiseLevel);
    particleTestList = generateGaussianData(nTestSize, noiseLevel);
  } else {
    particleList = generateSpiralData(nSize, noiseLevel);
    particleTestList = generateSpiralData(nTestSize, noiseLevel);
  }
  makeDataLabel();
}

function alphaColor(c, a) {
  var r = red(c);
  var g = green(c);
  var b = blue(c);
  return color(r, g, b, a);
}

// NEAT related code

function initModel() {

  var i, j;

  N.init({nInput: 2, nOutput: 1, initConfig: "all",
    activations : "default",
  });
  trainer = new N.NEATTrainer({
    new_node_rate : makeNodeProbability,
    new_connection_rate : makeConnectionProbability,
    sub_population_size : 20,
    init_weight_magnitude : 0.25,
    mutation_rate : 0.9,
    mutation_size : 0.005,
    extinction_rate : 0.5,
  });
  /*
  compressor = new N.NEATCompressor();
  */

  trainer.applyFitnessFunc(fitnessFunc);

  genome = trainer.getBestGenome();

  modelReady = true;
}

function evolveModel(g) {
  /*
  var i;
  for (i=0;i<1;i++) {
    if (Math.random() < 0.5) g.addRandomConnection();
    if (Math.random() < 0.5) g.addRandomNode();
  }
  */
}

function renderInfo(g) {
  if (typeof g === 'undefined') return;

  var text = "gen: "+N.getNumGeneration()+", nodes: "+g.getNodesInUse().length+",\t";
  text += "connections: "+g.connections.length+",\t";
  /*
  if (g.cluster) {
    text += "cluster: "+g.cluster+",\t";
  }
  */
  if (g.fitness) {
    text += "fitness: "+Math.round(10000*g.fitness,0)/10000+"<br/>";
  }
  if (presentationMode === false) {
    $("#drawGraph").html(text);
  }
}

function renderGraph(clusterNum_) {
  var genome;
  if (typeof clusterNum_ !== 'undefined') {
    genome = trainer.getBestGenome(clusterNum_);
  } else {
    genome = trainer.getBestGenome();
  }
  colaGraph = RenderGraph.getGenomeGraph(genome);
  //colaGraph = RenderGraph.getExampleGraph();
  renderInfo(genome);
  RenderGraph.drawGraph(colaGraph);
}

function setCluster(cluster) {
  var K = trainer.num_populations;
  var i;
  var c;

  for (i=0;i<K;i++) {
    if (i === cluster) {
      c = "rgba(0,136,204, 1.0)";
    } else {
      c = "rgba(0,136,204, 0.15)";
    }
    $("#cluster"+i).css('border-color',c);
  }
  if (typeof cluster !== 'undefined') {
    selectedCluster = cluster;
  }
  renderGraph(cluster);
  calculateAccuracy();
}

// p5 related code

var myCanvas;
function setup() {
  myCanvas = createCanvas(min($(window).width()*0.8, 640), min($(window).height()*0.6, 480));

  myCanvas.parent('p5Container');
  resizeCanvas(dataWidth+16, dataWidth+16);

  generateRandomData();
  initModel();

  frameRate(10);

  // at the beginning, evolve a few times
  for (var i=0;i<1;i++) {
    trainer.evolve();
    backprop(1);
  }

  renderGraph();

}

function drawDataPoint(p, prediction_) {
  var x, y;
  var s = 6;
  var prediction = typeof prediction_ === 'undefined' ? p.l : prediction_;
  x = p.x;
  y = p.y;
  l = p.l;
  if (prediction === l) { // correct label
    if (l === 0) {
      stroke(color(255, 165, 0, 192));
    } else {
      stroke(color(0, 165, 255, 192));
    }
  } else { // incorrect label
    stroke(color(255, 0, 0, 128));
  }
  if (l === 0) {
    fill(alphaColor(color(255, 165, 0), 128));
  } else {
    fill(alphaColor(color(0, 165, 255), 128));
  }
  ellipse(d2p(x), d2p(y), s, s);

}

var fitnessFunc = function(genome, _backpropMode, _nCycles) {
  // this function backprops the genome over nCycles on the dataset.
  // returns final error (fitness);
  "use strict";
  var i, j, m, n, y, t, e, avgError, finalError, initError, veryInitError;
  var G, output;
  var nCycles = 1;
  var backpropMode = false;
  n=particleList.length;

  function findTotalError() {
    var avgError = 0.0;
    genome.setupModel(n);
    genome.setInput(data);
    var G = new R.Graph(false);
    genome.forward(G);
    var output = genome.getOutput();
    output[0] = G.sigmoid(output[0]);

    for (i=0;i<n;i++) {
      y = output[0].w[i]; // prediction (not to be confused w/ coordinate)
      t = label.get(i, 0); // ground truth label
      e = -(t*Math.log(y)+(1-t)*Math.log(1-y)); // logistic regression
      avgError += e;
    }
    avgError /= n;
    return avgError;
  }

  if (_backpropMode === true) backpropMode = true;
  if (typeof _nCycles !== 'undefined') nCycles = _nCycles;

  // find the error first with no backprop
  initError = findTotalError();
  avgError = initError;
  veryInitError = initError;

  if (backpropMode === true) {

    var genomeBackup = genome.copy(); // make a copy, in case backprop messes shit up.
    var origGenomeBackup = genome.copy(); // make a copy, in case backprop REALLY messes shit up.

    for (j=0;j<nCycles;j++) {
      // try to find error
      avgError = 0.0;
      // make minibatch
      makeMiniBatch();
      genome.setupModel(nBatch);
      genome.setInput(dataBatch);
      G = new R.Graph();
      genome.forward(G);
      output = genome.getOutput();
      output[0] = G.sigmoid(output[0]);

      for (i=0;i<nBatch;i++) {
        y = output[0].w[i]; // prediction (not to be confused w/ coordinate)
        t = labelBatch.get(i, 0); // ground truth label
        e = -(t*Math.log(y)+(1-t)*Math.log(1-y)); // logistic regression
        output[0].dw[i] = (y-t) / nBatch;
        avgError += e/nBatch;
      }

      G.backward();
      solver.step(genome.model.connections, learnRate, 0.001, 5.0);
      genome.updateModelWeights();

      if (j > 0 && j % 20 === 0) {
        finalError = findTotalError();
        if (finalError > initError) {
          // if the final sumSqError is crappier than initSumSqError, just make genome return to initial guy.
          // console.log('leaving prematurely at j = '+j);
          genome.copyFrom(genomeBackup);
          break;
        } else {
          initError = finalError;
          genomeBackup = genome.copy();
        }
      }

    }

    avgError = findTotalError();
    if (avgError > initError) {
      avgError = initError;
      genome.copyFrom(genomeBackup);
    }
    if (avgError > veryInitError) {
      avgError = veryInitError;
      genome.copyFrom(origGenomeBackup);
      console.log('backprop was useless.');
    }

  }

  var penaltyNode = genome.getNodesInUse().length-3;
  var penaltyConnection = genome.connections.length;

  var penaltyFactor = 1 + penaltyNodeFactor*Math.sqrt(penaltyNode) + penaltyConnectionFactor*Math.sqrt(penaltyConnection);

  return -avgError*penaltyFactor;
};


function buildPredictionList(pList, thedata, thelabel, g, quantisation_) {
  // this function backprops the genome over nCycles on the dataset.
  // returns final error (fitness);
  "use strict";
  var i, n, y;
  var G, output;
  var acc = 0;
  var quantisation = typeof quantisation_ === 'undefined'? false : quantisation_;

  n=pList.length;

  g.setupModel(n);
  g.setInput(thedata);
  G = new R.Graph(false);
  g.forward(G);
  output = g.getOutput();
  output[0] = G.sigmoid(output[0]);

  for (i=0;i<n;i++) {
    y = output[0].w[i]; // prediction (not to be confused w/ coordinate)
    if (quantisation === false) {
      pList[i] = (y > 0.5)? 1.0: 0.0;
      acc += Math.round(y) === thelabel.get(i, 0)? 1 : 0;
    } else {
      pList[i] = y;
    }
  }

  acc /= n;

  return acc;

}

function draw() {
  var i, j, n;
  var fitness = 0.0;
  var theCluster = 0;
  var bestGenome;
  var dw = dataWidth/12;
  var remainOrange, remainBlue;

  noStroke();
  fill(255);
  rect(0, 0, width, height);

  textSize(8);
  textFont("Roboto");

  strokeWeight(0.5);
  stroke(64, 192);
  rect(1, 1, width-1-16, height-1-16);

  for (i=-6;i<=6;i++) {
    strokeWeight(0.5);
    stroke(64, 192);
    if (i > -6 && i < 6) {
      line((i+6)*dw+1, height-16, (i+6)*dw+1, height-12);
      strokeWeight(0);
      stroke(64, 192);
      fill(64, 192);
      text(""+(i === 0? "X" : i), (i+6)*dw-1, height-3);
    }
  }
  for (i=-6;i<=6;i++) {
    strokeWeight(0.5);
    stroke(64, 192);
    if (i > -6 && i < 6) {
      line(width-16, (i+6)*dw+1, width-12, (i+6)*dw+1);
      strokeWeight(0);
      stroke(64, 192);
      fill(64, 192);
      text(""+(i === 0? "Y" : -i), width-10, (i+6)*dw+3);
    }
  }

  if (modelReady) {

    image(img, 1, 1, dataWidth-1, dataWidth-1);

    if (showTrainData) {
      for (i=0;i<particleList.length;i++) {
        drawDataPoint(particleList[i], predictionList[i]);
      }
    }
    if (showTestData) {
      for (i=0;i<particleTestList.length;i++) {
        drawDataPoint(particleTestList[i], predictionTestList[i]);
      }
    }

    textSize(10);
    textFont("Courier New");

    /*
    stroke(200, 127, 0, 100);
    fill(240, 20, 20, 100);
    */
    stroke(0, 192);
    fill(0, 192);
    if (presentationMode === false) {
      if(desktopMode) {
        text("train accuracy = "+Math.round(accuracy*1000)/10+"%\ttest accuracy = "+Math.round(testAccuracy*1000)/10+"%", 8, height-10-14);
      } else {
        text("train accuracy = "+Math.round(accuracy*1000)/10+"%", 8, height-10-14);
      }
    }
  } else {
    // custom data mode
    remainOrange = requiredCustomData - customDataList[0].length;
    remainBlue = requiredCustomData - customDataList[1].length;
    textSize(10);
    textFont("Courier New");
    if (customDataMode === 0) {
      stroke(color(255, 165, 0, 128));
      fill(color(255, 165, 0, 128));
    } else {
      stroke(color(0, 165, 255, 128));
      fill(color(0, 165, 255, 128));
    }
    text("Tap in datapoints.\nThe more the better!", 8, 14);

    if (remainOrange > 0) {
      stroke(color(255, 165, 0, 128));
      fill(color(255, 165, 0, 128));
      text("Need "+remainOrange+" more orange datapoint"+(remainOrange>1?"s.":"."), 8, height-10-14-14);
    }
    if (remainBlue > 0) {
      stroke(color(0, 165, 255, 128));
      fill(color(0, 165, 255, 128));
      text("Need "+remainBlue+" more blue datapoint"+(remainOrange>1?"s.":"."), 8, height-10-14);
    }
    // draw datapoints
    for (j=0;j<2;j++) {
      for (i=0;i<customDataList[j].length;i++) {
        drawDataPoint(customDataList[j][i]);
      }
    }

  }

}

/*
var mousePressed = function() {
  devicePressed(mouseX, mouseY);
  return false;
};

var touchStarted = function() {
  devicePressed(touchX, touchY);
  return false;
};
*/

// use jquery for mouse events to avoid p5.js and jquery conflicts
$("#p5Container").click(function( event ) {

  //var rect = $("#p5Container").getBoundingClientRect();
  var pos = $("canvas:first").offset();
  var x = (event.pageX - pos.left - 0);
  var y = (event.pageY - pos.top - 0);

  devicePressed(x, y);
});

$("#spray_button").click(function() {
  generateRandomData();
  initModel();
  renderGraph();
//  generateRandomData();
});

$("#clear_button").click(function() {
  generateRandomData();
  initModel();
  renderGraph();
});

function colorClusters() {
  var K = trainer.num_populations;
  var i;
  var c, f;
  var fArray = R.zeros(K);
  var cArray = R.zeros(K);
  var best = -1e20;
  var worst = 1e20;
  var level;

  for (i=0;i<K;i++) {
    f = trainer.getBestGenome(i).fitness;
    best = Math.max(best, f);
    worst = Math.min(worst, f);
    fArray[i] = f;
  }

  for (i=0;i<K;i++) {
    f = trainer.getBestGenome(i).fitness;
    best = Math.max(best, f);
    worst = Math.min(worst, f);
    fArray[i] = f;
  }

  var range = Math.max(best-worst, 0.4);

  for (i=0;i<K;i++) {
    cArray[i] = (fArray[i]-worst)/range;
  }

  for (i=0;i<K;i++) {
    level = 0.15 + cArray[i] * 0.85;
    c = "rgba(0,136,204, "+level+")";
    $("#cluster"+i).css('background-color',c);
  }

}

function backprop(n,_clusterMode) {

  var clusterMode = true; // by default, we would cluster stuff (takes time)
  if (typeof _clusterMode !== 'undefined') {
    clusterMode = _clusterMode;
  }

  var f = function(g) {
    if (n > 1) {
      return fitnessFunc(g, true, n);
    }
    return fitnessFunc(g, false, 1);
  };
  trainer.applyFitnessFunc(f, clusterMode);
  genome = trainer.getBestGenome();
  if (typeof genome.cluster !== 'undefined') {
    selectedCluster = genome.cluster;
  } else {
    selectedCluster = -1;
  }
  setCluster(genome.cluster);
  colorClusters();
}

function createPredictionImage() {
  var i, j;
  var r, g, b, pred, dist;
  img = createImage(dataWidth/2-1, dataWidth/2-1);
  img.loadPixels();
  for (j = 0; j < img.height; j++) {
    for (i = 0; i < img.width; i++) {
      pred = imgPrediction[j*img.height+i];
      if (pred < 0.5) {
        stroke(color(255, 165, 0, 192));
        r = 255;
        g = 165;
        b = 0;
      } else {
        r = 0;
        g = 165;
        b = 255;
      }
      dist = Math.abs((pred-0.5)/0.5);
      img.set(i, j, color(r, g, b, 96*Math.abs(dist)));
    }
  }
  img.updatePixels();
}

function calculateAccuracy() {
  if (selectedCluster >= 0) {
    genome = trainer.getBestGenome(selectedCluster);
  } else {
    genome = trainer.getBestGenome();
  }
  bestGenome = trainer.getBestGenome();

  fitness = fitnessFunc(genome, false, 1);
  theCluster = genome.cluster;
  // draw best genome last
  genome=trainer.getBestGenome(theCluster);
  accuracy = buildPredictionList(predictionList, data, label, genome);
  testAccuracy = buildPredictionList(predictionTestList, testData, testLabel, genome);
  buildPredictionList(imgPrediction, imgData, null, genome, true);

  createPredictionImage();
}

$("#sgd_button").click(function() {
  $("#controlPanel").fadeOut(500, "swing", function() {
    $("#loadingSpinner").fadeIn(500, "swing", function() {
      backprop(nBackprop);
      $("#loadingSpinner").fadeOut(500, "swing", function() {
        $("#controlPanel").fadeIn(500, "swing");
      });
    });
  });
});

$("#evolve_button").click(function() {

  $("#controlPanel").fadeOut(500, "swing", function() {
    $("#loadingSpinner").fadeIn(500, "swing", function() {

      // beginning of callback hell
      trainer.evolve();

      // compress network:
      /*
      compressor.buildMap(trainer.getAllGenes());
      compressor.compressNEAT();
      compressor.compressGenes(trainer.genes);
      compressor.compressGenes(trainer.hallOfFame);
      compressor.compressGenes(trainer.bestOfSubPopulation);
      */

      // finished compression
      backprop(nBackprop);
      // end of callback hell

      $("#loadingSpinner").fadeOut(500, "swing", function() {
        $("#controlPanel").fadeIn(500, "swing");
      });
    });
  });

});

$("#cluster0").click(function() {
  setCluster(0);
});
$("#cluster1").click(function() {
  setCluster(1);
});
$("#cluster2").click(function() {
  setCluster(2);
});
$("#cluster3").click(function() {
  setCluster(3);
});
$("#cluster4").click(function() {
  setCluster(4);
});

$("#warning_button").click(function() {
  $("#warningText").hide();
});

$(function() {
    $( "#sliderNoise" ).slider({
      max : 0.99,
      min : 0.01,
      step : 0.01,
      value : noiseLevel,
      change: function (event, ui) {
        noiseLevel = ui.value;
        $("#noiseLevel").html("data noise level = "+noiseLevel);
        if (dataSetChoice <= 3) {
          generateRandomData();
        }
      },
    });
});

$(function() {
    $( "#sliderNode" ).slider({
      max : 0.2,
      step : 0.005,
      value : penaltyNodeFactor,
      change: function (event, ui) {
        penaltyNodeFactor = ui.value;
        $("#penaltyNode").html("node count penalty = "+penaltyNodeFactor);
      },
    });
});

$(function() {
    $( "#sliderConnection" ).slider({
      max : 0.2,
      step : 0.005,
      value : penaltyConnectionFactor,
      change: function (event, ui) {
        penaltyConnectionFactor = ui.value;
        $("#penaltyConnection").html("connection count penalty = "+penaltyConnectionFactor);
      },
    });
});

$("#noiseLevel").html("data noise level = "+noiseLevel);
$("#penaltyNode").html("node count penalty = "+penaltyNodeFactor);
$("#penaltyConnection").html("connection count penalty = "+penaltyConnectionFactor);

$(function() {
    $( "#sliderBackprop" ).slider({
      max : 1200,
      min : 100,
      step : 50,
      value : nBackprop,
      change: function (event, ui) {
        nBackprop = ui.value;
        $("#backpropDisplay").html("backprop steps = "+nBackprop);
      },
    });
});
$("#backpropDisplay").html("backprop steps = "+nBackprop);

$(function() {
    $( "#sliderLearnRate" ).slider({
      max : 4,
      min : 0,
      step : .01,
      value : 3,
      change: function (event, ui) {
        learnRate = Math.round(Math.pow(10, -(5-ui.value))*100000)/100000;
        $("#learnRateDisplay").html("learning rate = "+learnRate);
      },
    });
});
$("#learnRateDisplay").html("learning rate = "+learnRate);

// below crappily written speghetti code is for custom data sets:
var customDataMode = R.randi(0, 2); // 0 for orange, 1 for blue
var customDataList = [[], []];
function colorCustomDataChoice() {
  var c0, c1;
  if (customDataMode === 0) {
    c0 = "rgba(255, 165, 0, 0.9)";
    c1 = "rgba(0, 165, 255, 0.4)";
  } else {
    c0 = "rgba(255, 165, 0, 0.4)";
    c1 = "rgba(0, 165, 255, 0.9)";
  }
  $("#customDataOrange").css('border-color',"rgba(255, 165, 0, 1.0)");
  $("#customDataBlue").css('border-color',"rgba(0, 165, 255, 1.0)");
  $("#customDataOrange").css('background-color',c0);
  $("#customDataBlue").css('background-color',c1);
}
function getCustomData() {
  modelReady = false;
  customDataList = [[], []];
  colorCustomDataChoice();
  $("#customDataSubmit").css('border-color', "rgba(81,163,81, 1.0");
  $("#customDataSubmit").css('background-color', "rgba(81,163,81, 0.4");
  $("#customDataBox").show();
  $("#controlPanel").hide();
}
$("#customDataOrange").click(function() {
  customDataMode = 0;
  colorCustomDataChoice();
});
$("#customDataBlue").click(function() {
  customDataMode = 1;
  colorCustomDataChoice();
});
$("#customDataSubmit").click(function(){
  if (customDataList[0].length >= requiredCustomData && customDataList[0].length >= requiredCustomData) {
    shuffleParticleList(customDataList[0]);
    shuffleParticleList(customDataList[1]);
    var orangeTestIndex =　Math.floor(customDataList[0].length/2);
    var blueTestIndex =　Math.floor(customDataList[0].length/2);
    var i;
    particleList = customDataList[0].slice(0, orangeTestIndex);
    particleList = particleList.concat(customDataList[1].slice(0, orangeTestIndex));
    particleTestList = customDataList[0].slice(orangeTestIndex);
    particleTestList = particleTestList.concat(customDataList[1].slice(orangeTestIndex));
    shuffleParticleList(particleList);
    shuffleParticleList(particleTestList);
    nSize = particleList.length;
    nTestSize = particleTestList.length;
    $("#customDataBox").hide();
    $("#controlPanel").show();
    makeDataLabel();
    initModel();
    // at the beginning, evolve a few times
    for (var i=0;i<1;i++) {
      trainer.evolve();
      backprop(1);
    }
    renderGraph();

  }
});

function recordTrainingData(x, y) {
  p = new Particle(x, y, customDataMode);
  customDataList[customDataMode].push(p);
  if (customDataList[0].length >= requiredCustomData && customDataList[0].length >= requiredCustomData) {
    $("#customDataSubmit").css('border-color', "rgba(81,163,81, 1.0");
    $("#customDataSubmit").css('background-color', "rgba(81,163,81, 0.9");
  }
}

// When the mouse is pressed we. . .
var devicePressed = function(x, y) {
  // put point in the queue to train
  if (x < 0 || y < 0 || x > (width-16) || y > (height-16)) return;
  recordTrainingData(p2d(x), p2d(y));
};

$(function() {
  $( "#dataChoiceMode" ).change( function (event) {
      var theChoice = event.target.selectedIndex;
      if (theChoice <= 3) {
        dataSetChoice = theChoice;
        generateRandomData();
        initModel();
        // at the beginning, evolve a few times
        for (var i=0;i<1;i++) {
          trainer.evolve();
          backprop(1);
        }
        renderGraph();
      } else if (theChoice === 4) { // custom data
        dataSetChoice = theChoice;
        getCustomData();
      } else { // reset
        $("#dataChoiceMode")[0].selectedIndex = dataSetChoice;
        if (dataSetChoice <= 3) {
          generateRandomData();
          calculateAccuracy();
        } else {
          getCustomData();
        }
      }
  });

  $( "#dataDisplayMode" ).change( function (event) {
    var displayMode = event.target.selectedIndex;
    if (displayMode === 0) {
      showTrainData = true;
      showTestData = false;
    } else if (displayMode === 1) {
      showTestData = true;
      showTrainData = false;
    } else {
      showTrainData = true;
      showTestData = true;
    }
  });
});

$("#loadingSpinner").hide();

