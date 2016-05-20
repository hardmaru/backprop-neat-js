// MIT License

// heavily modified recurrent.js library for use in genetic art
// with DCT compression, CoSyNe neuroevolution
// sin, cos, gaussian, abs activations

// based off https://github.com/karpathy/recurrentjs, excellent library by karpathy

var R = {};

(function(global) {
  "use strict";

  // Utility fun
  function assert(condition, message) {
    // from http://stackoverflow.com/questions/15313418/javascript-assert
    if (!condition) {
      message = message || "Assertion failed";
      if (typeof Error !== "undefined") {
        throw new Error(message);
      }
      throw message; // Fallback
    }
  }

  // Random numbers utils
  var return_v = false;
  var v_val = 0.0;
  var gaussRandom = function() {
    if(return_v) {
      return_v = false;
      return v_val;
    }
    var u = 2*Math.random()-1;
    var v = 2*Math.random()-1;
    var r = u*u + v*v;
    if(r === 0 || r > 1) return gaussRandom();
    var c = Math.sqrt(-2*Math.log(r)/r);
    v_val = v*c; // cache this
    return_v = true;
    return u*c;
  };
  var randf = function(a, b) { return Math.random()*(b-a)+a; };
  var randi = function(a, b) { return Math.floor(Math.random()*(b-a)+a); };
  var randn = function(mu, std){ return mu+gaussRandom()*std; };

  // helper function returns array of zeros of length n
  // and uses typed arrays if available
  var zeros = function(n) {
    if(typeof(n)==='undefined' || isNaN(n)) { return []; }
    if(typeof ArrayBuffer === 'undefined') {
      // lacking browser support
      var arr = new Array(n);
      for(var i=0;i<n;i++) { arr[i] = 0; }
      return arr;
    } else {
      if (n > 150000) console.log('creating a float array of length = '+n);
      //return new Float64Array(n);
      return new Float32Array(n);
    }
  };

  var copy = function(floatArray) {
    // returns a copy of floatArray
    var n = floatArray.length;
    var result = zeros(n);
    for (var i=0;i<n;i++) {
      result[i]=floatArray[i];
    }
    return result;
  };

  var shuffle = function(origArray) {
    // returns a newArray which is a shuffled version of origArray
    var i, randomIndex;
    var temp;
    var N = origArray.length;
    var result = zeros(N);
    for (i=0;i<N;i++) {
      result[i] = origArray[i];
    }
    for (i=0;i<N;i++) {
      // swaps i with randomIndex
      randomIndex = randi(0, N);
      temp = result[randomIndex];
      result[randomIndex] = result[i];
      result[i] = temp;
    }
    return result;
  };

  // Mat holds a matrix
  var Mat = function(n,d) {
    // n is number of rows d is number of columns
    this.n = n;
    this.d = d;
    this.w = zeros(n * d);
    this.dw = zeros(n * d);
  };
  Mat.prototype = {
    get: function(row, col) {
      // slow but careful accessor function
      // we want row-major order
      var ix = (this.d * row) + col;
      assert(ix >= 0 && ix < this.w.length);
      return this.w[ix];
    },
    set: function(row, col, v) {
      // slow but careful accessor function
      var ix = (this.d * row) + col;
      assert(ix >= 0 && ix < this.w.length);
      this.w[ix] = v;
    },
    setAll: function(v) {
      // sets all value of Mat (.w) to v
      var i, n;
      for (i=0,n=this.n*this.d;i<n;i++) {
        this.w[i] = v;
      }
    },
    setFromArray: function(a) {
      var i, j;
      assert(this.n === a.length && this.d === a[0].length);
      for (i=0;i<this.n;i++) {
        for (j=0;j<this.d;j++) {
          this.set(i, j, a[i][j]);
        }
      }
    },
    copy: function() {
      // return a copy of Mat
      var result = new Mat(this.n, this.d);
      var i, len;
      len = this.n*this.d;
      for (i = 0; i < len; i++) {
        result.w[i] = this.w[i];
        result.dw[i] = this.dw[i];
      }
      return result;
    },
    toString: function(precision_) {
      var result_w = '[';
      var i, j;
      var n, d;
      var ix;
      var precision = 10e-4 || precision_;
      precision = 1/precision;
      n = this.n;
      d = this.d;
      for (i=0;i<n;i++) {
        for (j=0;j<d;j++) {
          ix = i*d+j;
          assert(ix >= 0 && ix < this.w.length);
          result_w+=''+Math.round(precision*this.w[ix])/precision+',\t';
        }
        result_w+='\n';
      }
      result_w+=']';
      return result_w;
    },
    print: function() {
      console.log(this.toString());
    },
    dct2: function() {
      // inefficient implementation of discrete cosine transform (2d)
      // ref: http://www.mathworks.com/help/images/ref/dct2.html
      var n = this.n;
      var d = this.d;
      var B = new Mat(n, d); // resulting matrix
      var i, j, k, l;
      var temp;
      for (i=0;i<n;i++) {
        for (j=0;j<d;j++) {
          temp=0;
          for (k=0;k<n;k++) {
            for (l=0;l<d;l++) {
              temp=temp+this.w[k*d+l]*Math.cos(i*Math.PI*((2*k)+1)/(2*n))*Math.cos(j*Math.PI*((2*l)+1)/(2*d));
            }
          }
          if ((i===0)&&(j!==0)) temp*=1/Math.SQRT2;
          if ((j===0)&&(i!==0)) temp*=1/Math.SQRT2;
          if ((j===0)&&(i===0)) temp*=0.5;
          B.w[i*d+j]=temp*2/Math.sqrt(n*d);
        }
      }
      return B;
    },
    idct2: function() {
      // inefficient implementation of inverse discrete cosine transform (2d)
      // ref: http://www.mathworks.com/help/images/ref/idct2.html
      var n = this.n;
      var d = this.d;
      var A = new Mat(n, d); // resulting matrix
      var i, j, k, l;
      var temp;
      for (i=0;i<n;i++) {
        for (j=0;j<d;j++) {
          A.w[i*d+j]=0;
          for (k=0;k<n;k++) {
            for (l=0;l<d;l++) {
              temp=this.w[k*d+l]*Math.cos((((2*i)+1)*k*Math.PI)/(2*n))*Math.cos((((2*j)+1)*l*Math.PI)/(2*d));

              if ((k===0)&&(l===0)) temp*=0.5;
              if ((k!==0)&&(l===0)) temp*=1/Math.SQRT2;
              if ((k===0)&&(l!==0)) temp*=1/Math.SQRT2;

              A.w[i*d+j]+=temp*2/Math.sqrt(n*d);

            }
          }
        }
      }
      return A;
    },
    toJSON: function() {
      var json = {};
      json.n = this.n;
      json.d = this.d;
      json.w = this.w;
      return json;
    },
    fromJSON: function(json) {
      this.n = json.n;
      this.d = json.d;
      this.w = zeros(this.n * this.d);
      this.dw = zeros(this.n * this.d);
      for(var i=0,n=this.n * this.d;i<n;i++) {
        this.w[i] = json.w[i]; // copy over weights
      }
    }
  };

  // return Mat but filled with random numbers from gaussian
  var RandMat = function(n,d,mu,std) {
    var m = new Mat(n, d);
    fillRandn(m,mu || 0,std || 0.08); // kind of :P
    return m;
  };

  // Mat utils
  // fill matrix with random gaussian numbers
  var fillRandn = function(m, mu, std) { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randn(mu, std); } };
  var fillRand = function(m, lo, hi) { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randf(lo, hi); } };

  // Transformer definitions
  var Graph = function(needs_backprop) {
    if(typeof needs_backprop === 'undefined') { needs_backprop = true; }
    this.needs_backprop = needs_backprop;

    // this will store a list of functions that perform backprop,
    // in their forward pass order. So in backprop we will go
    // backwards and evoke each one
    this.backprop = [];
  };
  Graph.prototype = {
    backward: function() {
      for(var i=this.backprop.length-1;i>=0;i--) {
        this.backprop[i](); // tick!
      }
    },
    rowPluck: function(m, ix) {
      // pluck a row of m with index ix and return it as col vector
      assert(ix >= 0 && ix < m.n);
      var d = m.d;
      var out = new Mat(d, 1);
      for(var i=0,n=d;i<n;i++){ out.w[i] = m.w[d * ix + i]; } // copy over the data

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0,n=d;i<n;i++){ m.dw[d * ix + i] += out.dw[i]; }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    sin: function(m) {
      // tanh nonlinearity
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) {
        out.w[i] = Math.sin(m.w[i]);
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            // grad for z = sin(x) is cos(x)
            var mwi = out.w[i];
            m.dw[i] += Math.cos(m.w[i]) * out.dw[i];
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    cos: function(m) {
      // tanh nonlinearity
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) {
        out.w[i] = Math.cos(m.w[i]);
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            // grad for z = sin(x) is cos(x)
            var mwi = out.w[i];
            m.dw[i] += -Math.sin(m.w[i]) * out.dw[i];
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    gaussian: function(m) {
      // tanh nonlinearity
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      //var c = (1.0/Math.sqrt(2*Math.PI)); // constant of 1 / sqrt(2*pi)
      var c = 1.0; // make amplitude bigger than normal gaussian
      for(var i=0;i<n;i++) {
        out.w[i] = c*Math.exp(-m.w[i]*m.w[i]/2);
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            // grad for z = sin(x) is cos(x)
            var mwi = out.w[i];
            m.dw[i] += -m.w[i] * mwi * out.dw[i];
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    tanh: function(m) {
      // tanh nonlinearity
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) {
        out.w[i] = Math.tanh(m.w[i]);
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            // grad for z = tanh(x) is (1 - z^2)
            var mwi = out.w[i];
            m.dw[i] += (1.0 - mwi * mwi) * out.dw[i];
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    sigmoid: function(m) {
      // sigmoid nonlinearity
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) {
        out.w[i] = sig(m.w[i]);
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            // grad for z = sigmoid(x) is z(1 - z)
            var mwi = out.w[i];
            m.dw[i] += mwi * (1.0 - mwi) * out.dw[i];
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    relu: function(m) {
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) {
        out.w[i] = Math.max(0, m.w[i]); // relu
      }
      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            m.dw[i] += m.w[i] > 0 ? out.dw[i] : 0.0;
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    abs: function(m) {
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) {
        out.w[i] = Math.abs(m.w[i]); // relu
      }
      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            m.dw[i] += m.w[i] > 0 ? out.dw[i] : -out.dw[i];
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    mul: function(m1, m2) {
      // multiply matrices m1 * m2
      assert(m1.d === m2.n, 'matmul dimensions misaligned');

      var n = m1.n;
      var d = m2.d;
      var out = new Mat(n,d);
      for(var i=0;i<m1.n;i++) { // loop over rows of m1
        for(var j=0;j<m2.d;j++) { // loop over cols of m2
          var dot = 0.0;
          for(var k=0;k<m1.d;k++) { // dot product loop
            dot += m1.w[m1.d*i+k] * m2.w[m2.d*k+j];
          }
          out.w[d*i+j] = dot;
        }
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<m1.n;i++) { // loop over rows of m1
            for(var j=0;j<m2.d;j++) { // loop over cols of m2
              for(var k=0;k<m1.d;k++) { // dot product loop
                var b = out.dw[d*i+j];
                m1.dw[m1.d*i+k] += m2.w[m2.d*k+j] * b;
                m2.dw[m2.d*k+j] += m1.w[m1.d*i+k] * b;
              }
            }
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    add: function(m1, m2) {
      assert(m1.w.length === m2.w.length);

      var out = new Mat(m1.n, m1.d);
      for(var i=0,n=m1.w.length;i<n;i++) {
        out.w[i] = m1.w[i] + m2.w[i];
      }
      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0,n=m1.w.length;i<n;i++) {
            m1.dw[i] += out.dw[i];
            m2.dw[i] += out.dw[i];
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    eltmul: function(m1, m2) {
      assert(m1.w.length === m2.w.length);

      var out = new Mat(m1.n, m1.d);
      for(var i=0,n=m1.w.length;i<n;i++) {
        out.w[i] = m1.w[i] * m2.w[i];
      }
      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0,n=m1.w.length;i<n;i++) {
            m1.dw[i] += m2.w[i] * out.dw[i];
            m2.dw[i] += m1.w[i] * out.dw[i];
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
  };

  var softmax = function(m) {
      var out = new Mat(m.n, m.d); // probability volume
      var maxval = -999999;
      var i, n;
      for(i=0,n=m.w.length;i<n;i++) { if(m.w[i] > maxval) maxval = m.w[i]; }

      var s = 0.0;
      for(i=0,n=m.w.length;i<n;i++) {
        out.w[i] = Math.exp(m.w[i] - maxval);
        s += out.w[i];
      }
      for(i=0,n=m.w.length;i<n;i++) { out.w[i] /= s; }

      // no backward pass here needed
      // since we will use the computed probabilities outside
      // to set gradients directly on m
      return out;
  };

  var Solver = function() {
    this.decay_rate = 0.999;
    this.smooth_eps = 1e-8;
    this.step_cache = {};
  };
  Solver.prototype = {
    step: function(model, step_size, regc, clipval) {
      // perform parameter update
      var solver_stats = {};
      var num_clipped = 0;
      var num_tot = 0;
      for(var k in model) {
        if(model.hasOwnProperty(k)) {
          var m = model[k]; // mat ref
          if(!(k in this.step_cache)) { this.step_cache[k] = new Mat(m.n, m.d); }
          var s = this.step_cache[k];
          for(var i=0,n=m.w.length;i<n;i++) {

            // rmsprop adaptive learning rate
            var mdwi = m.dw[i];
            if (isNaN(mdwi)) {
              /*
              console.log('backprop has numerical issues.');
              console.log('dWeight '+i+' is NaN');
              console.log('setting dw to zero');
              */
              m.dw[i] = 0.0;
              mdwi = 0.0;
            }
            s.w[i] = s.w[i] * this.decay_rate + (1.0 - this.decay_rate) * mdwi * mdwi;

            // gradient clip
            if(mdwi > clipval) {
              mdwi = clipval;
              num_clipped++;
            }
            if(mdwi < -clipval) {
              mdwi = -clipval;
              num_clipped++;
            }
            num_tot++;

            if ((s.w[i] + this.smooth_eps) <= 0) {
              console.log('rmsprop has numerical issues');
              console.log('step_cache '+i+' = '+s.w[i]);
              console.log('smooth_eps = '+this.smooth_eps);
            }

            // update (and regularize)
            m.w[i] += - step_size * mdwi / Math.sqrt(Math.max(s.w[i],this.smooth_eps)) - regc * m.w[i];
            m.dw[i] = 0; // reset gradients for next iteration

            // clip the actual weights as well
            if(m.w[i] > clipval*10) {
              //console.log('rmsprop clipped the weight with orig value '+m.w[i]);
              m.w[i] = clipval*10;
            } else if(m.w[i] < -clipval*10) {
              //console.log('rmsprop clipped the weight with orig value '+m.w[i]);
              m.w[i] = -clipval*10;
            }

            assert(!isNaN(m.w[i]), 'weight '+i+' is NaN');

          }
        }
      }
      solver_stats.ratio_clipped = num_clipped*1.0/num_tot;
      return solver_stats;
    }
  };

  var initLSTM = function(input_size, hidden_sizes, output_size) {
    // hidden size should be a list

    var model = {};
    var hidden_size;
    var prev_size;
    for(var d=0;d<hidden_sizes.length;d++) { // loop over depths
      prev_size = d === 0 ? input_size : hidden_sizes[d - 1];
      hidden_size = hidden_sizes[d];

      // gates parameters
      model['Wix'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);
      model['Wih'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
      model['bi'+d] = new Mat(hidden_size, 1);
      model['Wfx'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);
      model['Wfh'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
      model['bf'+d] = new Mat(hidden_size, 1);
      model['Wox'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);
      model['Woh'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
      model['bo'+d] = new Mat(hidden_size, 1);
      // cell write params
      model['Wcx'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);
      model['Wch'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
      model['bc'+d] = new Mat(hidden_size, 1);
    }
    // decoder params
    model.Whd = new RandMat(output_size, hidden_size, 0, 0.08);
    model.bd = new Mat(output_size, 1);
    return model;
  };

  var forwardLSTM = function(G, model, hidden_sizes, x, prev) {
    // forward prop for a single tick of LSTM
    // G is graph to append ops to
    // model contains LSTM parameters
    // x is 1D column vector with observation
    // prev is a struct containing hidden and cell
    // from previous iteration

    var hidden_prevs;
    var cell_prevs;
    var d;

    if(typeof prev.h === 'undefined') {
      hidden_prevs = [];
      cell_prevs = [];
      for(d=0;d<hidden_sizes.length;d++) {
        hidden_prevs.push(new R.Mat(hidden_sizes[d],1));
        cell_prevs.push(new R.Mat(hidden_sizes[d],1));
      }
    } else {
      hidden_prevs = prev.h;
      cell_prevs = prev.c;
    }

    var hidden = [];
    var cell = [];
    for(d=0;d<hidden_sizes.length;d++) {

      var input_vector = d === 0 ? x : hidden[d-1];
      var hidden_prev = hidden_prevs[d];
      var cell_prev = cell_prevs[d];

      // input gate
      var h0 = G.mul(model['Wix'+d], input_vector);
      var h1 = G.mul(model['Wih'+d], hidden_prev);
      var input_gate = G.sigmoid(G.add(G.add(h0,h1),model['bi'+d]));

      // forget gate
      var h2 = G.mul(model['Wfx'+d], input_vector);
      var h3 = G.mul(model['Wfh'+d], hidden_prev);
      var forget_gate = G.sigmoid(G.add(G.add(h2, h3),model['bf'+d]));

      // output gate
      var h4 = G.mul(model['Wox'+d], input_vector);
      var h5 = G.mul(model['Woh'+d], hidden_prev);
      var output_gate = G.sigmoid(G.add(G.add(h4, h5),model['bo'+d]));

      // write operation on cells
      var h6 = G.mul(model['Wcx'+d], input_vector);
      var h7 = G.mul(model['Wch'+d], hidden_prev);
      var cell_write = G.tanh(G.add(G.add(h6, h7),model['bc'+d]));

      // compute new cell activation
      var retain_cell = G.eltmul(forget_gate, cell_prev); // what do we keep from cell
      var write_cell = G.eltmul(input_gate, cell_write); // what do we write to cell
      var cell_d = G.add(retain_cell, write_cell); // new cell contents

      // compute hidden state as gated, saturated cell activations
      var hidden_d = G.eltmul(output_gate, G.tanh(cell_d));

      hidden.push(hidden_d);
      cell.push(cell_d);
    }

    // one decoder to outputs at end
    var output = G.add(G.mul(model.Whd, hidden[hidden.length - 1]),model.bd);

    // return cell memory, hidden representation and output
    return {'h':hidden, 'c':cell, 'o' : output};
  };

  var initRNN = function(input_size, hidden_sizes, output_size) {
    // hidden size should be a list

    var model = {};
    var hidden_size, prev_size;
    for(var d=0;d<hidden_sizes.length;d++) { // loop over depths
      prev_size = d === 0 ? input_size : hidden_sizes[d - 1];
      hidden_size = hidden_sizes[d];
      model['Wxh'+d] = new R.RandMat(hidden_size, prev_size , 0, 0.08);
      model['Whh'+d] = new R.RandMat(hidden_size, hidden_size, 0, 0.08);
      model['bhh'+d] = new R.Mat(hidden_size, 1);
    }
    // decoder params
    model.Whd = new RandMat(output_size, hidden_size, 0, 0.08);
    model.bd = new Mat(output_size, 1);
    return model;
  };

  var forwardRNN = function(G, model, hidden_sizes, x, prev) {
    // forward prop for a single tick of RNN
    // G is graph to append ops to
    // model contains RNN parameters
    // x is 1D column vector with observation
    // prev is a struct containing hidden activations from last step

    var hidden_prevs;
    var d;
    if(typeof prev.h === 'undefined') {
      hidden_prevs = [];
      for(d=0;d<hidden_sizes.length;d++) {
        hidden_prevs.push(new R.Mat(hidden_sizes[d],1));
      }
    } else {
      hidden_prevs = prev.h;
    }

    var hidden = [];
    for(d=0;d<hidden_sizes.length;d++) {

      var input_vector = d === 0 ? x : hidden[d-1];
      var hidden_prev = hidden_prevs[d];

      var h0 = G.mul(model['Wxh'+d], input_vector);
      var h1 = G.mul(model['Whh'+d], hidden_prev);
      var hidden_d = G.relu(G.add(G.add(h0, h1), model['bhh'+d]));

      hidden.push(hidden_d);
    }

    // one decoder to outputs at end
    var output = G.add(G.mul(model.Whd, hidden[hidden.length - 1]),model.bd);

    // return cell memory, hidden representation and output
    return {'h':hidden, 'o' : output};
  };

  var sig = function(x) {
    // helper function for computing sigmoid
    return 1.0/(1+Math.exp(-x));
  };

  var maxi = function(w) {
    // argmax of array w
    var maxv = w[0];
    var maxix = 0;
    for(var i=1,n=w.length;i<n;i++) {
      var v = w[i];
      if(v > maxv) {
        maxix = i;
        maxv = v;
      }
    }
    return maxix;
  };

  var samplei = function(w) {
    // sample argmax from w, assuming w are
    // probabilities that sum to one
    var r = randf(0,1);
    var x = 0.0;
    var i = 0;
    while(true) {
      x += w[i];
      if(x > r) { return i; }
      i++;
    }
    return w.length - 1; // pretty sure we should never get here?
  };

  var getModelSize = function(model) {
    // returns the size (ie, number of floats) used in a model
    var len = 0;
    var k;
    var m;
    for(k in model) {
      if(model.hasOwnProperty(k)) {
        m = model[k];
        len += m.w.length;
      }
    }
    return len;
  };

  // other utils
  var flattenModel = function(model, gradient_) {
    // returns an float array containing a dump of model's params in order
    // if gradient_ is true, the the flatten model returns the dw's, rather whan w's.
    var len = 0; // determine length of dump
    var i = 0;
    var j;
    var k;
    var m;
    var gradientMode = false || gradient_;
    len = getModelSize(model);
    var result = R.zeros(len);
    for(k in model) {
      if(model.hasOwnProperty(k)) {
        m = model[k];
        len = m.w.length;
        for (j = 0; j < len; j++) {
          if (gradientMode) {
            result[i] = m.dw[j];
          } else {
            result[i] = m.w[j];
          }
          i++;
        }
      }
    }
    return result;
  };
  var pushToModel = function(model, dump, gradient_) {
    // pushes a float array containing a dump of model's params into a model
    // if gradient_ is true, dump will be pushed to dw's, rather whan w's.
    var len = 0; // determine length of dump
    var i = 0;
    var j;
    var k;
    var m;
    var gradientMode = false || gradient_;
    len = getModelSize(model);
    assert(dump.length === len); // make sure the array dump has same len as model
    for(k in model) {
      if(model.hasOwnProperty(k)) {
        m = model[k];
        len = m.w.length;
        for (j = 0; j < len; j++) {
          if (gradientMode) {
            m.dw[j] = dump[i];
          } else {
            m.w[j] = dump[i];
          }
          i++;
        }
      }
    }
  };
  var copyModel = function(model) {
    // returns an exact copy of a model
    var k;
    var m;
    var result = [];
    for(k in model) {
      if(model.hasOwnProperty(k)) {
        m = model[k];
        result[k] = m.copy();
      }
    }
    return result;
  };
  var zeroOutModel = function(model, gradientAsWell_) {
    // zeros out every element (including dw, if gradient_ is true) of model
    var len = 0; // determine length of dump
    var j;
    var k;
    var m;
    var gradientMode = false || gradientAsWell_;

    for(k in model) {
      if(model.hasOwnProperty(k)) {
        m = model[k];
        len = m.w.length;
        for (j = 0; j < len; j++) {
          if (gradientMode) {
            m.dw[j] = 0;
          }
          m.w[j] = 0;
        }
      }
    }
  };
  var compressModel = function(model, nCoef) {
    // returns a compressed model using 2d-dct
    // each model param will be compressed down to:
    // min(nRow, nCoef), min(nCol, nCoef)
    var k;
    var m;
    var nRow, nCol;
    var result = [];
    var z; // dct transformed matrix
    var i, j;
    for(k in model) {
      if(model.hasOwnProperty(k)) {
        m = model[k];
        z = m.dct2();
        nRow = Math.min(z.n, nCoef);
        nCol = Math.min(z.d, nCoef);
        result[k] = new Mat(nRow, nCol);
        for (i=0;i<nRow;i++) {
          for (j=0;j<nCol;j++) {
            result[k].set(i, j, z.get(i, j));
          }
        }
      }
    }
    return result;
  };
  var decompressModel = function(small, model) {
    // decompresses small (a compressed model) into model using idct
    var k;
    var m, s;
    var nRow, nCol;
    var z; // idct transformed matrix
    var i, j;
    for(k in small) {
      if(small.hasOwnProperty(k)) {
        s = small[k];
        m = model[k];
        nRow = m.n;
        nCol = m.d;
        z = new Mat(nRow, nCol);
        for (i=0;i<s.n;i++) {
          for (j=0;j<s.d;j++) {
            z.set(i, j, s.get(i, j));
          }
        }
        model[k] = z.idct2();
      }
    }
  };
  var numGradient = function(f, model, avgDiff_, epsilon_) {
    // calculates numerical gradient.  fitness f is forward pass function passed in is a function of model only.
    // f will be run many times when the params of each indiv weight changes
    // numerical gradient is computed off average of uptick and downtick gradient, so O(e^2) noise.
    // returns a mat object, where .w holds percentage differences, and .dw holds numerical gradient
    // if avgDiff_ mode is set to true, returns the average percentage diff rather than the actual gradients
    var epsilon = 1e-10 || epsilon_;
    var avgDiff = false || avgDiff_;
    var base = f(model);
    var upBase, downBase;
    var uptick = copyModel(model); // uptick.w holds the ticked weight value
    var downtick = copyModel(model); // opposite of uptick.w
    var numGrad = copyModel(model); // numGrad.dw holds the numerical gradient.

    var avgPercentDiff = 0.0;
    var avgPercentDiffCounter = 0;

    var i, len;
    var m;
    var k;
    var result = [];
    for(k in model) {
      if(model.hasOwnProperty(k)) {
        m = model[k];
        len = m.w.length;
        for (i = 0; i < len; i++) {
          // change the weights by a small amount to find gradient
          uptick[k].w[i] += epsilon;
          downtick[k].w[i] -= epsilon;
          upBase = f(uptick);
          downBase = f(downtick);
          // store numerical gradient
          numGrad[k].dw[i] = (upBase - downBase) / (2 * epsilon);
          numGrad[k].w[i] = ((numGrad[k].dw[i] + epsilon) / (model[k].dw[i] + epsilon) - 1);
          avgPercentDiff += numGrad[k].w[i] * numGrad[k].w[i];
          avgPercentDiffCounter += 1;
          // store precentage diff in w.
          // undo the change of weights by a small amount
          uptick[k].w[i] -= epsilon;
          downtick[k].w[i] += epsilon;

          // set model's dw to numerical gradient (useful for debugging)
          //model[k].dw[i] = numGrad[k].dw[i];
        }
      }
    }

    if (avgDiff) {
      return Math.sqrt(avgPercentDiff / avgPercentDiffCounter);
    }
    return numGrad;
  };

  // neuroevolution tools

  // chromosome implementation using an array of floats
  var Gene = function(initFloatArray) {
    var i;
    var len = initFloatArray.length;
    // the input array will be copied
    this.fitness = -1e20; // default fitness value is very negative
    this.nTrial = 0; // number of trials subjected to so far.
    this.gene = zeros(len);
    for (i=0;i<len;i++) {
      this.gene[i] = initFloatArray[i];
    }
  };

  Gene.prototype = {
    burstMutate: function(burst_magnitude_) { // adds a normal random variable of stdev width, zero mean to each gene.
      var burst_magnitude = burst_magnitude_ || 0.1;
      var i, N;
      N = this.gene.length;
      for (i = 0; i < N; i++) {
        this.gene[i] += randn(0.0, burst_magnitude);
      }
    },
    randomize: function(burst_magnitude_) { // resets each gene to a random value with zero mean and stdev
      var burst_magnitude = burst_magnitude_ || 0.1;
      var i, N;
      N = this.gene.length;
      for (i = 0; i < N; i++) {
        this.gene[i] = randn(0.0, burst_magnitude);
      }
    },
    mutate: function(mutation_rate_, burst_magnitude_) { // adds random gaussian (0,stdev) to each gene with prob mutation_rate
      var mutation_rate = mutation_rate_ || 0.1;
      var burst_magnitude = burst_magnitude_ || 0.1;
      var i, N;
      N = this.gene.length;
      for (i = 0; i < N; i++) {
        if (randf(0,1) < mutation_rate) {
          this.gene[i] += randn(0.0, burst_magnitude);
        }
      }
    },
    crossover: function(partner, kid1, kid2, onePoint) {
      // performs one-point crossover with partner to produce 2 kids
      // edit -> changed to uniform crossover method.
      // assumes all chromosomes are initialised with same array size. pls make sure of this before calling
      assert(this.gene.length === partner.gene.length);
      assert(partner.gene.length === kid1.gene.length);
      assert(kid1.gene.length === kid2.gene.length);
      var onePointMode = false;
      if (typeof onePoint !== 'undefined') onePointMode = onePoint;
      var i, N;
      N = this.gene.length;
      var cross = true;
      var l = randi(0, N); // crossover point (for one point xover)
      for (i = 0; i < N; i++) {
        if (onePointMode) {
          cross = (i < l);
        } else {
          cross = (Math.random() < 0.5);
        }
        if (cross) {
          kid1.gene[i] = this.gene[i];
          kid2.gene[i] = partner.gene[i];
        } else {
          kid1.gene[i] = partner.gene[i];
          kid2.gene[i] = this.gene[i];
        }
      }
    },
    copyFrom: function(g) { // copies g's gene into itself
      var i, N;
      this.copyFromArray(g.gene);
    },
    copyFromArray: function(sourceFloatArray) {
      // copy an array own's gene (must be same size)
      assert(this.gene.length === sourceFloatArray.length);
      var i, N;
      N = this.gene.length;
      for (i = 0; i < N; i++) {
        this.gene[i] = sourceFloatArray[i];
      }
    },
    copy: function(precision_) { // returns a rounded exact copy of itself (into new memory, doesn't return reference)
      var precision = 10e-4 || precision_; precision = 1/precision;
      var newFloatArray = zeros(this.gene.length);
      var i;
      for (i = 0; i < this.gene.length; i++) {
        newFloatArray[i] = Math.round(precision*this.gene[i])/precision;
      }
      var g = new Gene(newFloatArray);
      g.fitness = this.fitness;
      g.nTrial = this.nTrial;
      return g;
    },
    pushToModel: function(model) { // pushes this chromosome to a specified network
      pushToModel(model, this.gene);
    }
  };

  // randomize neural network model with random weights and biases
  var randomizeModel = function(model, magnitude_) {
    var modelSize = getModelSize(model);
    var magnitude = 1.0 || magnitude_;
    var r = new RandMat(1, modelSize, 0, magnitude);
    var g = new Gene(r.w);
    g.pushToModel(model);
  };

  var GATrainer = function(model, options_, init_gene_array) {
    // implementation of CoSyNe neuroevolution framework
    //
    // options:
    // population_size : positive integer
    // hall_of_fame_size : positive integer, stores best guys in all of history and keeps them.
    // mutation_rate : [0, 1], when mutation happens, chance of each gene getting mutated
    // elite_percentage : [0, 0.3], only this group mates and produces offsprings
    // mutation_size : positive floating point.  stdev of gausian noise added for mutations
    // target_fitness : after fitness achieved is greater than this float value, learning stops
    // init_weight_magnitude : stdev of initial random weight (default = 1.0)
    // burst_generations : positive integer.  if best fitness doesn't improve after this number of generations
    //                    then mutate everything!
    // best_trial : default 1.  save best of best_trial's results for each chromosome.
    // num_match : for use in arms race mode.  how many random matches we set for each chromosome when it is its turn.
    //
    // init_gene_array:  init float array to initialize the chromosomes.  can be result obtained from pretrained sessions.
    // debug_mode: false by default.  if set to true, console.log debug output occurs.

    this.model = copyModel(model); // makes a local working copy of the model. copies architecture and weights

    var options = options_ || {};
    this.hall_of_fame_size = typeof options.hall_of_fame_size !== 'undefined' ? options.hall_of_fame_size : 5;
    this.population_size = typeof options.population_size !== 'undefined' ? options.population_size : 30;
    this.population_size += this.hall_of_fame_size; // make room for hall of fame beyond specified population size.
    this.population_size = Math.floor(this.population_size/2)*2; // make sure even number
    this.length = this.population_size; // use the variable length to suit array pattern.
    this.mutation_rate = typeof options.mutation_rate !== 'undefined' ? options.mutation_rate : 0.01;
    this.init_weight_magnitude = typeof options.init_weight_magnitude !== 'undefined' ? options.init_weight_magnitude : 1.0;
    this.elite_percentage = typeof options.elite_percentage !== 'undefined' ? options.elite_percentage : 0.2;
    this.mutation_size = typeof options.mutation_size !== 'undefined' ? options.mutation_size : 0.5;
    this.debug_mode = typeof options.debug_mode !== 'undefined' ? options.debug_mode : false;
    this.gene_size = getModelSize(this.model); // number of floats in each gene

    var initGene;
    var i;
    var gene;
    if (init_gene_array) {
      initGene = new Gene(init_gene_array);
    }

    this.genes = []; // population
    this.hallOfFame = []; // stores the hall of fame here!
    for (i = 0; i < this.population_size; i++) {
      gene = new Gene(zeros(this.gene_size));
      if (initGene) { // if initial gene supplied, burst mutate param.
        gene.copyFrom(initGene);
        if (i > 0) { // don't mutate the first guy.
          gene.burstMutate(this.mutation_size);
        }
      } else {
        gene.randomize(this.init_weight_magnitude);
      }
      this.genes.push(gene);
    }
    // generates first few hall of fame genes (but burst mutates some of them)
    for (i = 0; i < this.hall_of_fame_size; i++) {
      gene = new Gene(zeros(this.gene_size));
      if (init_gene_array) { // if initial gene supplied, burst mutate param.
        gene.copyFrom(initGene);
      } else {
        gene.randomize(this.init_weight_magnitude);
        if (i > 0) { // don't mutate the first guy.
          gene.burstMutate(this.mutation_size);
        }
      }
      this.hallOfFame.push(gene);
    }

    pushToModel(this.model, this.genes[0].gene); // push first chromosome to neural network. (replaced *1 above)

  };

  GATrainer.prototype = {
    sortByFitness: function(c) {
      c = c.sort(function (a, b) {
        if (a.fitness > b.fitness) { return -1; }
        if (a.fitness < b.fitness) { return 1; }
        return 0;
      });
    },
    pushGeneToModel: function(model, i) {
      // pushes the i th gene of the sorted population into a model
      // this ignores hall of fame
      var g = this.genes[i];
      g.pushToModel(model);
    },
    pushBestGeneToModel: function(model) {
      this.pushGeneToModel(model, 0);
    },
    pushHistToModel: function(model, i) {
      // pushes the i th gene of the sorted population into a model from the hall-of-fame
      // this requires hall of fame model to be used
      var Nh = this.hall_of_fame_size;
      assert(Nh > 0); // hall of fame must be used.
      var g = this.hallOfFame[i];
      g.pushToModel(model);
    },
    pushBestHistToModel: function(model) {
      this.pushHistToModel(model, 0);
    },
    flushFitness: function() {
      // resets all the fitness scores to very negative numbers, incl hall-of-fame
      var i, N, Nh;
      var c = this.genes;
      var h = this.hallOfFame;
      N = this.population_size;
      Nh = this.hall_of_fame_size;
      for (i=0;i<N;i++) {
        c[i].fitness = -1e20;
      }
      for (i=0;i<Nh;i++) {
        h[i].fitness = -1e20;
      }

    },
    sortGenes: function() {
      // this functions just sort list of genes by fitness and does not do any
      // cross over or mutations.
      var c = this.genes;
      // sort the chromosomes by fitness
      this.sortByFitness(c);
    },
    evolve: function() {
      // this function does bare minimum simulation of one generation
      // it assumes the code prior to calling evolve would have simulated the system
      // it also assumes that the fitness in each chromosome of this trainer will have been assigned
      // it just does the task of crossovers and mutations afterwards.

      var i, j, N, Nh;
      var c = this.genes;
      var h = this.hallOfFame;

      N = this.population_size;
      Nh = this.hall_of_fame_size;

      // sort the chromosomes by fitness
      this.sortByFitness(c);

      if (this.debug_mode) {
        for (i = 0; i < 5; i++) {
          console.log(i+': '+Math.round(c[i].fitness*100)/100);
        }
        for (i = 5; i >= 1; i--) {
          console.log((N-i)+': '+Math.round(c[N-i].fitness*100)/100);
        }
      }

      // copies best from population to hall of fame:
      for (i = 0; i < Nh; i++) {
        h.push(c[i].copy());
      }

      // sorts hall of fame
      this.sortByFitness(h);
      // cuts off hall of fame to keep only Nh elements
      h = h.slice(0, Nh);

      if (this.debug_mode) {
        console.log('hall of fame:');
        for (i = 0; i < Math.min(Nh, 3); i++) {
          console.log(i+': '+Math.round(h[i].fitness*100)/100);
        }
      }

      // alters population:

      var Nelite = Math.floor(Math.floor(this.elite_percentage*N)/2)*2; // even number
      for (i = Nelite; i < N; i+=2) {
        var p1 = randi(0, Nelite);
        var p2 = randi(0, Nelite);
        c[p1].crossover(c[p2], c[i], c[i+1]);
      }

      // leaves the last Nh slots for hall of fame guys.
      for (i = 0; i < N-Nh; i++) {
        c[i].mutate(this.mutation_rate, this.mutation_size);
      }

      // sneakily puts in the hall of famers back into the population at the end:
      for (i = 0; i < Nh; i++) {
        c[N-Nh+i] = h[i].copy();
      }


      // permutation step in CoSyNe
      // we permute all weights in elite set, and don't prob-weight as in Gomez 2008.

      var Nperm = Nelite; // permute the weights up to Nperm.
      var permData = zeros(Nperm);
      var len = c[0].gene.length; // number of elements in each gene
      for (j=0;j<len;j++) {
        // populate the data to be shuffled
        for (i=0;i<Nperm;i++) {
          permData[i] = c[i].gene[j];
        }
        permData = shuffle(permData); // the magic is supposed to happen here.
        // put back the shuffled data back:
        for (i=0;i<Nperm;i++) {
          c[i].gene[j] = permData[i];
        }
      }

    }
  };

  // various utils
  global.maxi = maxi;
  global.samplei = samplei;
  global.randi = randi;
  global.randf = randf;
  global.randn = randn;
  global.zeros = zeros;
  global.copy = copy;
  global.shuffle = shuffle;
  global.softmax = softmax;
  global.assert = assert;

  // classes
  global.Mat = Mat;
  global.RandMat = RandMat;

  global.forwardLSTM = forwardLSTM;
  global.initLSTM = initLSTM;
  global.forwardRNN = forwardRNN;
  global.initRNN = initRNN;

  // optimization
  global.Solver = Solver;
  global.Graph = Graph;

  // other utils
  global.flattenModel = flattenModel;
  global.getModelSize = getModelSize;
  global.copyModel = copyModel;
  global.zeroOutModel = zeroOutModel;
  global.numGradient = numGradient;
  global.pushToModel = pushToModel;
  global.randomizeModel = randomizeModel;

  // model compression
  global.compressModel = compressModel;
  global.decompressModel = decompressModel;

  // ga
  global.GATrainer = GATrainer;
  global.Gene = Gene;

})(R);
(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    //window.jsfeat = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(R);
