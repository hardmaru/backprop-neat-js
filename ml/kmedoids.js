/*globals paper, console, $ */
/*jslint nomen: true, undef: true, sloppy: true */

// kmedoids library

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

// implementation of neat algorithm with recurrent.js graphs to support backprop
// used for genetic art.
// code is not modular or oop, ad done in a fortran/scientific computing style
// apologies in advance if that ain't ur taste.
if (typeof module != "undefined") {
  var R = require('./recurrent.js');
  //var NetArt = require('./netart.js');
}

var kmedoids = {};

(function(global) {
  "use strict";

  // Utility fun
  var assert = function(condition, message) {
    // from http://stackoverflow.com/questions/15313418/javascript-assert
    if (!condition) {
      message = message || "Assertion failed";
      if (typeof Error !== "undefined") {
        throw new Error(message);
      }
      throw message; // Fallback
    }
  };

  var cluster =[];
  var nCluster = 1;
  var LARGENUM = 10e20;

  var initCluster = function() {
    var i;
    cluster = [];
    for (i=0;i<nCluster;i++) {
      cluster.push([]);
    }
  };

  var init = function(nCluster_) {
    nCluster = nCluster_;
    initCluster();
  };

  var shuffle = function(origArray) {
    // returns a newArray which is a shuffled version of origArray
    var i, randomIndex;
    var temp;
    var N = origArray.length;
    var result = new Array(N);
    for (i=0;i<N;i++) {
      result[i] = origArray[i];
    }
    for (i=0;i<N;i++) {
      // swaps i with randomIndex
      randomIndex = R.randi(0, N);
      temp = result[randomIndex];
      result[randomIndex] = result[i];
      result[i] = temp;
    }
    return result;
  };

  var copyArray = function(origArray) {
    // returns a newArray which is a copy of origArray
    var i;
    var N = origArray.length;
    var result = new Array(N);
    for (i=0;i<N;i++) {
      result[i] = origArray[i];
    }
    return result;
  };

  var equalArray = function(array1, array2) {
    // returns true if two arrays are equal
    if (array1.length !== array2.length) return false;
    var i;
    var N = array1.length;
    for (i=0;i<N;i++) {
      if (array1[i] !== array2[i]) return false;
    }
    return true;
  };

  var swapArray = function(array, idx1, idx2) {
    // swap array[idx1] and array[idx2]
    var temp = array[idx1];
    array[idx1] = array[idx2];
    array[idx2] = temp;
  };

  var dist = function(e1, e2) { // function can be overloaded to other dist func
    var dx = e2.x-e1.x;
    var dy = e2.y-e1.y;
    return (dx*dx+dy*dy); //removed sqr root
  };

  function clusterLength() {
    var i, j, m, n;
    var count=0;
    for (i=0,m=cluster.length;i<m;i++) {
      for (j=0,n=cluster[i].length;j<n;j++) {
        count++;
      }
    }
    return count;
  }

  var lloyd_partition = function(list) {
    var i, j, m, n, idx, d, d2, k, cost, bestCost, anchor;

    n = list.length;
    var distTable = new R.Mat(n, n);

    for (i=0;i<n;i++) {
      for (j=0;j<=i;j++) {
        if (i===j) {
          distTable.set(i, j, 0);
        }
        d = dist(list[i], list[j]);
        distTable.set(i, j, d);
        distTable.set(j, i, d);
      }
    }

    var idxArray = new Array(n);
    for (i=0;i<n;i++) {
      idxArray[i] = i;
    }
    idxArray = shuffle(idxArray);

    var anchorArray = idxArray.splice(0, nCluster);

    var oldAnchorArray;

    var maxTry = 100;
    var countTry = 0;

    do {

      oldAnchorArray = copyArray(anchorArray);
      initCluster(); // wipes out cluster

      // push anchor array into clusters first as the first element
      for (j=0;j<nCluster;j++) {
        cluster[j].push(anchorArray[j]);
      }

      // go thru remaining idx Arrays to assign each element to closest anchor
      for (i=0,n=idxArray.length;i<n;i++) {
        k=-1;
        d=LARGENUM;
        for (j=0;j<nCluster;j++) {
          d2 = distTable.get(idxArray[i], anchorArray[j]);
          if (d2 < d) {
            k = j;
            d = d2;
          }
        }
        assert(k>=0, 'cannot find closest distance to index; all distances greater than '+LARGENUM);
        cluster[k].push(idxArray[i]);
      }

      // for each cluster, reassign the anchor position
      for (i=0;i<nCluster;i++) {
        anchor=-1;
        bestCost=LARGENUM;
        n = cluster[i].length;
        for (j=0;j<n;j++) {
          cost = 0;
          for (k=0;k<n;k++) {
            cost += distTable.get(cluster[i][j],cluster[i][k]);
          }
          if (cost < bestCost) {
            bestCost = cost;
            anchor = j;
          }
        }
        assert(anchor>=0, 'cannot find a good anchor position');
        swapArray(cluster[i], 0, anchor);
      }

      // reprocess the clusters back into array and repeat until it converges
      idxArray = [];
      for (i=0;i<nCluster;i++) {
        anchorArray[i] = cluster[i][0];
        n = cluster[i].length;
        for (j=1;j<n;j++) {
          idxArray.push(cluster[i][j]);
        }
      }

      countTry++;

      if (countTry >= maxTry) {
        // console.log('k-medoids did not converge after '+maxTry+ ' tries.');
        break;
      }

    } while(!equalArray(anchorArray, oldAnchorArray));

  };

  var pam_partition = function(list) {
    var i, j, m, n, idx, d, d2, k, cost, bestCost, anchor, temp;

    bestCost = LARGENUM;

    n = list.length;
    var distTable = new R.Mat(n, n);

    for (i=0;i<n;i++) {
      for (j=0;j<=i;j++) {
        if (i===j) {
          distTable.set(i, j, 0);
        }
        d = dist(list[i], list[j]);
        distTable.set(i, j, d);
        distTable.set(j, i, d);
      }
    }

    var idxArray = new Array(n);
    for (i=0;i<n;i++) {
      idxArray[i] = i;
    }
    idxArray = shuffle(idxArray);

    var anchorArray = idxArray.splice(0, nCluster);

    var oldAnchorArray;

    var maxTry = 100;
    var countTry = 0;

    function buildCluster() {
      var i, j, k, n, d, d2;
      var localCost = 0;
      initCluster(); // wipes out cluster

      // push anchor array into clusters first as the first element
      for (j=0;j<nCluster;j++) {
        cluster[j].push(anchorArray[j]);
      }

      // go thru remaining idx Arrays to assign each element to closest anchor
      for (i=0,n=idxArray.length;i<n;i++) {
        k=-1;
        d=LARGENUM;
        for (j=0;j<nCluster;j++) {
          d2 = distTable.get(idxArray[i], anchorArray[j]);
          if (d2 < d) {
            k = j;
            d = d2;
          }
        }
        assert(k>=0, 'cannot build cluster since all distances from anchor is greater than '+LARGENUM);
        cluster[k].push(idxArray[i]);
        localCost += d;
      }
      return localCost;
    }

    do {

      oldAnchorArray = copyArray(anchorArray);

      bestCost = buildCluster();

      for (i=0;i<nCluster;i++) {
        for (j=0,n=idxArray.length;j<n;j++) {
          // swap
          temp = anchorArray[i];
          anchorArray[i] = idxArray[j];
          idxArray[j] = temp;

          cost = buildCluster();

          // swap back if it doesn't work
          if (cost > bestCost) {
            temp = anchorArray[i];
            anchorArray[i] = idxArray[j];
            idxArray[j] = temp;
          } else {
            bestCost = cost;
          }
        }
      }

      bestCost = buildCluster();
      //console.log('best cost = '+bestCost);

      countTry++;

      if (countTry >= maxTry) {
        // console.log('k-medoids did not converge after '+maxTry+ ' tries.');
        break;
      }

    } while(!equalArray(anchorArray, oldAnchorArray));

  };

  var partition = pam_partition;

  var getCluster = function() {
    return cluster;
  };

  var pushToCluster = function(elementIdx, idx_) {
    var idx = 0;
    if (typeof idx_ !== 'undefined') idx = idx_;
    cluster[idx].push(elementIdx);
  };

  global.init = init;
  global.setDistFunction = function(distFunc) {
    dist = distFunc;
  };
  global.getCluster = getCluster;
  global.partition = partition;
  global.pushToCluster = pushToCluster;

})(kmedoids);
(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    //window.jsfeat = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(kmedoids);


