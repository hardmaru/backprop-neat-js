/*globals paper, console, $ */
/*jslint nomen: true, undef: true, sloppy: true */

// network art library

/*

@licstart  The following is the entire license notice for the
JavaScript code in this page.

Copyright (C) 2015 david ha, otoro.net, otoro labs

The JavaScript code in this page is free software: you can
redistribute it and/or modify it under the terms of the GNU
General Public License (GNU GPL) as published by the Free Software
Foundation, either version 3 of the License, or (at your option)
any later version.  The code is distributed WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU GPL for more details.

As additional permission under GNU GPL version 3 section 7, you
may distribute non-source (e.g., minimized or compacted) forms of
that code without the copy of the GNU GPL normally required by
section 4, provided you include this license notice and a URL
through which recipients can access the Corresponding Source.


@licend  The above is the entire license notice
for the JavaScript code in this page.
*/

// this module draws constrained graphs in the webbrowser

// assumes that recurrent.js and neat.js are preloaded

// this module cannot be run in node.js

var RenderGraph = {};

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
  var NODE_MULT = 9;

  var NODE_HIDDEN_RANGE_LO = 3;
  var NODE_HIDDEN_RANGE_HI = 9;

/*
  var operators = [null, null, null, 'sigmoid', 'tanh', 'relu', 'gaussian', 'sin', 'cos', 'abs', 'mult', 'add', 'mult', 'add'];

  // for connections
  var IDX_CONNECTION = 0;
  var IDX_WEIGHT = 1;
  var IDX_ACTIVE = 2;

  //var activations = [NODE_SIGMOID, NODE_TANH, NODE_RELU, NODE_GAUSSIAN, NODE_SIN, NODE_COS, NODE_MULT, NODE_ABS, NODE_ADD, NODE_MGAUSSIAN, NODE_SQUARE];
  var activations = [NODE_SIGMOID, NODE_TANH, NODE_RELU, NODE_GAUSSIAN, NODE_SIN, NODE_COS, NODE_ADD];
  */

  var colorTable = [
      '#2124FF', // input
      '#FF2718', // output
      '#1F22C1', // bias
      '#EE8A2A', // sigmoid
      '#B17516', // tanh
      '#B1B0AA', // relu
      '#2CB11F', // gaussian
      '#F6DE39', // sin
      '#C5B12C', // cos
      '#E685E7',  // absolute value
      '#257580',  // multiplication
      '#68C6D3',  // addition
      '#3E8727',  // multiplicative gaussian
      '#F050E6'  // square
  ];

  var nameTable = [
      'input', // input
      'output', // output
      'bias', // bias
      'sigmoid', // sigmoid
      'tanh', // tanh
      'relu', // relu
      'gaussian', // gaussian
      'sine', // sin
      'cosine', // cos
      'abs',  // absolute value
      'mult',  // multiplication
      'add',  // addition
      'gaussian mult',  // multiplicative gaussian
      'square'  // square
  ];

  var width = 400,
      height = 320;

  var repelForce = 6;

  var d3cola;

  var svg;
  svg = d3.select("#drawGraph").append("svg:svg")
      .attr("width", width)
      .attr("height", height);

  var removeSVG = function() {
    d3.select("svg")
           .remove();
    svg.selectAll("*").remove();
  };

  var initSVG = function(domName_) {

    var domName = domName_ || "#drawGraph";

    d3.select("svg")
           .remove();
    svg.selectAll("*").remove();

    d3cola = cola.d3adaptor()
          .avoidOverlaps(true)
          .size([width, height]);

    svg = d3.select(domName).append("svg:svg")
      .attr("width", width)
      .attr("height", height);
    // define arrow markers for graph links
    svg.append('svg:defs').append('svg:marker')
        .attr('id', 'end-arrow')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 6)
        .attr('markerWidth', 3*2)
        .attr('markerHeight', 3*2)
        .attr('orient', 'auto')
        .append('svg:path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', '#000');
  };

  var drawGraph = function(graph, domName_) {
      var nodeRadius = desktopMode? 4*1.5 : 3*1.5;

      var domName = domName_ || "#drawGraph";

      initSVG(domName);

      function isIE() { return ((navigator.appName == 'Microsoft Internet Explorer') || ((navigator.appName == 'Netscape') && (new RegExp("Trident/.*rv:([0-9]{1,}[\.0-9]{0,})").exec(navigator.userAgent) !== null))); }

      graph.nodes.forEach(function (v) {
        if (v.name < 3) { // if bias or input or output, double the size.
          v.height = v.width = 2 * nodeRadius * 2;
        } else {
          v.height = v.width = 2 * nodeRadius;
        }
      });

      //console.log('graphing, number of nodes = '+graph.nodes.length+', number of connections = '+graph.links.length);

      if (graph.nodes.length>300 ) {
        // too slow, quit.
        // || graph.links.length === 0 // or no connections, quit
        removeSVG();
        return;
      }

      d3cola
          .nodes(graph.nodes)
          .links(graph.links)
          .symmetricDiffLinkLengths(repelForce)
          .constraints(graph.constraints)
          .start(50, 100, 200);

      var path = svg.selectAll(".link")
          .data(graph.links)
          .enter().append('svg:path')
          .attr('class', 'link')
          .style("stroke-width", function (d) { return Math.max(0.2,Math.log(Math.abs(d.weight)+1.0)); }); // Math.log(Math.abs(d.flows[0]))

      var margin = 0.25, pad = 0.25;

      var node = svg.selectAll(".node")
          .data(graph.nodes)
          .enter().append("circle")
          .attr("class", "node")
          .attr("r", function(d) {
                if (d.active) {
                  if (d.name < 3) {
                    return nodeRadius;
                  }
                  return nodeRadius / 1.25;
                }
                return 1;
            })
          .attr("label", "x")
          .style("fill", function (d) {
                if (d.active === 0) {
                    return 'rgba(255, 255, 255, 0)';
                }
                return colorTable[d.name];
            })
          .call(d3cola.drag);

      node.append("title")
          .text(function (d) { return nameTable[d.name]; });

      var label = svg.selectAll(".label")
          .data(graph.nodes)
          .enter().append("text")
          .attr("class", "label")
          .text(function (d) {
            if (nameTable[d.name] === 'input' || nameTable[d.name] === 'output' || nameTable[d.name] === 'bias') {
              return nameTable[d.name];
            }
            return nameTable[d.name];
          })
          .call(d3cola.drag)
          .each(function (d) {
              var b = this.getBBox();
              var extra = 2 * margin + 2 * pad;
              d.width = b.width + extra;
              d.height = b.height + extra;
          });


      d3cola.on("tick", function () {
          path.each(function (d) {
              if (isIE()) this.parentNode.insertBefore(this, this);
          });
          // draw directed edges with proper padding from node centers
          path.attr('d', function (d) {
              var r = 1;
              if (d.target.name < 3) {
                r = nodeRadius;
              } else {
                r = nodeRadius / 1.5;
              }
              var deltaX = d.target.x - d.source.x,
                  deltaY = d.target.y - d.source.y,
                  dist = Math.sqrt(deltaX * deltaX + deltaY * deltaY),
                  normX = deltaX / dist,
                  normY = deltaY / dist,
                  sourcePadding = r,
                  targetPadding = r + 2,
                  sourceX = d.source.x + (sourcePadding * normX),
                  sourceY = d.source.y + (sourcePadding * normY),
                  targetX = d.target.x - (targetPadding * normX),
                  targetY = d.target.y - (targetPadding * normY);
              return 'M' + sourceX + ',' + sourceY + 'L' + targetX + ',' + targetY;
          });

          node.attr("cx", function (d) { return d.x; })
              .attr("cy", function (d) { return d.y; });


          label
              .attr("x", function (d) {
                if (d.name === 0) {
                  return d.x;
                } else if (d.name === 1) {
                  return d.x;
                } else if (d.name === 2) {
                  //return d.x-18;
                  return d.x;
                }
                return d.x;
              })
              .attr("y", function (d) {
                var s = 5;
                if (d.name === 0) {
                  return d.y + (2 + 2) / 2 + 15;
                } else if (d.name === 1) {
                  return d.y - (2 + 2) / 2 - 8;
                } else if (d.name === 2) {
                  //return d.y+4.5;
                  return d.y + (2 + 2) / 2 + 15;
                }
                return d.y + (2 + 2) / 2 + 15;
              });

      });
  };

  var getExampleGraph = function() {
    var g = {
        nodes:[
            {name:0, active:1},
            {name:1, active:1},
            {name:2, active:1},
            {name:3, active:1},
            {name:4, active:1},
            {name:5, active:1},
            {name:6, active:1},
            {name:7, active:1},
            {name:8, active:1}
          ],
          links:[
            {source:0,target:6, weight:R.randn(0, 2)},
            {source:1,target:7, weight:R.randn(0, 2)},
            {source:0,target:2, weight:R.randn(0, 2)},
            {source:2,target:6, weight:R.randn(0, 2)},
            {source:2,target:8, weight:R.randn(0, 2)},
            {source:2,target:3, weight:R.randn(0, 2)},
            {source:7,target:3, weight:R.randn(0, 2)},
            {source:3,target:8, weight:R.randn(0, 2)},
            {source:1,target:3, weight:R.randn(0, 2)},
            {source:0,target:4, weight:R.randn(0, 2)},
            {source:1,target:4, weight:R.randn(0, 2)},
            {source:0,target:5, weight:R.randn(0, 2)},
            {source:1,target:5, weight:R.randn(0, 2)},
            {source:4,target:8, weight:R.randn(0, 2)},
            {source:4,target:5, weight:R.randn(0, 2)},
            {source:5,target:6, weight:R.randn(0, 2)},
            {source:8,target:5, weight:R.randn(0, 2)},
            {source:0,target:8, weight:R.randn(0, 2)},
            {source:6,target:8, weight:R.randn(0, 2)},
            {source:7,target:8, weight:R.randn(0, 2)}
          ],
        constraints:[
          {type:"alignment",
           axis:"x",
           offsets:[
             {node:0, offset:0},
             {node:1, offset:250},
             {node:8, offset:125}
           ]},
          {type:"alignment",
           axis:"y",
           offsets:[
             {node:0, offset:0},
             {node:1, offset:0},
             {node:8, offset:-450}
           ]}
        ]
    };
    return g;
  };

  var getGenomeGraph = function(genome) {
    var nInput = N.getNumInput();
    var nOutput = N.getNumOutput();
    var nodes = N.getNodes();
    var len = nodes.length;
    var connections = N.getConnections();
    var i, n;
    var cIndex;
    var outIndex;

    var widthOffsets = [];
    var heightOffsets = [];

    var activeNodes = R.zeros(nodes.length);

    for (i=0,n=genome.connections.length;i<n;i++) {
        cIndex = genome.connections[i][0];
        activeNodes[connections[cIndex][0]] = 1;
        activeNodes[connections[cIndex][1]] = 1;
    }

    // make inputs, bias and output nodes active even if they are not connected

    for (i=0,n=nInput+1+nOutput;i<n;i++) {
      activeNodes[i] = 1;
    }

    var maxWidth = width * 0.8;
    var maxHeight = height * 0.8;

    var g = {
        nodes:[],
        links:[],
        constraints:[]
    };

    var indexDict = R.zeros(nodes.length); // translate into graph's index from node index
    var count = 0;

    for (i=0, n=len;i<n;i++) {
      //g.nodes.push({name:nodes[i],active:activeNodes[i]});
      if (activeNodes[i] === 1 ) { // only push active nodes
        // || (i < nInput+1) || (i >= len-2-nOutput) // or if input, bias, or output nodes.
        g.nodes.push({name:nodes[i],active:activeNodes[i]});
        indexDict[i] = count;
        count++;
      }
    }


    for (i=0, n=genome.connections.length;i<n;i++) {
        cIndex = genome.connections[i][0];
        if (genome.connections[i][2]===1) { // if connection is enabled.
            g.links.push({source:indexDict[connections[cIndex][0]],target:indexDict[connections[cIndex][1]],
              weight:genome.connections[i][1]});
        }
    }

    var factor = 0.9;
    //for (i=0; i < (nInput+1); i++) {
    for (i=nInput; i >=0; i--) {
        widthOffsets.push({
            node: i,
            offset: (i+1) * maxWidth / (nInput+0),
        });
        heightOffsets.push({
            node: i,
            offset: maxHeight/1.75,
        });
    }

    for (i=0; i < (nOutput)+0; i++) {
        outIndex = nInput+1+i;
        widthOffsets.push({
            node: outIndex,
            offset: (i+1) * maxWidth / (nOutput+1),
        });
        heightOffsets.push({
            node: outIndex,
            offset: -maxHeight/3,
        });
    }

    // middle init weight
/*
    widthOffsets.push({
      node: outIndex+nOutput,
      offset: 0
    });
    heightOffsets.push({
      node: outIndex+nOutput,
      offset: -factor*maxHeight*0.5
    });
*/

    g.constraints.push(
        {type:"alignment",
           axis:"y",
           offsets: heightOffsets
       }
    );

    g.constraints.push(
        {type:"alignment",
           axis:"x",
           offsets: widthOffsets
       }
    );
/*
    console.log(maxWidth);
    console.log(maxHeight);
    console.log(g.constraints);
*/
    return g;

  };

  global.getExampleGraph = getExampleGraph;
  global.removeSVG = removeSVG;
  global.drawGraph = drawGraph;
  global.getGenomeGraph = getGenomeGraph;

})(RenderGraph);
(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    window.jsfeat = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(RenderGraph);


