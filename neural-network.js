'use strict'


var Matrix = require('node-matrix');
var sigmoid = require('sigmoid');

var sigmoidPrime =  function(z) {
  return Math.exp(-z) / Math.pow(1 + Math.exp(-z), 2);
}


module.exports = StravaNN;

function StravaNN(options) {
  if (!(this instanceof StravaNN)) return new StravaNN(options);
  options = options || {};

    this.activate = sigmoid;
		this.activatePrime = sigmoidPrime;

  // If these aren't set then use the numbers below
  this.learningRate = options.learningRate || 0.7;
  this.iterations = options.iterations || 10000;
  this.hiddenUnits = options.hiddenUnits || 3;
}




/*=================================

Forward Propagation

===================================*/

StravaNN.prototype.forward = function(examples) {
	//from the defaults
  var activate = this.activate;
  var weights = this.weights;

	//set new object
  var obj = {};

	//multiply the inputs & weights
  obj.hiddenSum = Matrix.multiply(weights.inputHidden, examples.input);
	//activate with the activation function defined
  obj.hiddenResult = obj.hiddenSum.transform(activate);

	//multiply the hidden layer & weights
  obj.outputSum = Matrix.multiply(weights.hiddenOutput, obj.hiddenResult);
	//activate hidden layer with function defined
  obj.outputResult = obj.outputSum.transform(activate);

	return obj;
};




/*=================================

Backward Propagation

===================================*/

StravaNN.prototype.back = function(examples, results) {
  var activatePrime = this.activatePrime;
  var learningRate = this.learningRate;
  var weights = this.weights;


  // compute weight adjustments
  var errorOutputLayer = Matrix.subtract(examples.output, results.outputResult);
  var deltaOutputLayer = Matrix.multiplyElements(results.outputSum.transform(activatePrime), errorOutputLayer);
  var hiddenOutputChanges = Matrix.multiplyScalar(Matrix.multiply(deltaOutputLayer, results.hiddenResult.transpose()), learningRate);
  var deltaHiddenLayer = Matrix.multiplyElements(Matrix.multiply(weights.hiddenOutput.transpose(), deltaOutputLayer), results.hiddenSum.transform(activatePrime));
  var inputHiddenChanges = Matrix.multiplyScalar(Matrix.multiply(deltaHiddenLayer, examples.input.transpose()), learningRate);

  // adjust weights
  weights.inputHidden = Matrix.add(weights.inputHidden, inputHiddenChanges);
  weights.hiddenOutput = Matrix.add(weights.hiddenOutput, hiddenOutputChanges);

  return errorOutputLayer;
};



/*=================================

Learn

===================================*/

StravaNN.prototype.learn = function(examples) {
 examples = normalize(examples);


  this.weights = {
    inputHidden: Matrix({
      columns: this.hiddenUnits,
      rows: examples.input[0].length,
      values: sample
    }),
    hiddenOutput: Matrix({
      columns: examples.output[0].length,
      rows: this.hiddenUnits,
      values: sample
    })
  };


  for (var i = 0; i < this.iterations; i++) {
    var results = this.forward(examples);
    var errors = this.back(examples, results);
  }

  return this;
};




/*=================================

Normalize the data

===================================*/

function normalize(data) {
  var ret = { input: [], output: [] };


  for (var i = 0; i < data.length; i++) {
    var datum = data[i];

    ret.output.push(datum.output);
    ret.input.push(datum.input);
  }

  ret.output = Matrix(ret.output);
  ret.input = Matrix(ret.input);


  return ret;

}


/*=================================

Predict

===================================*/


StravaNN.prototype.predict = function(input) {
  var results = this.forward({ input: Matrix([input]) });
	//console.log(results.length)


  return results.outputResult;
};



/*=================================

Assign Random number as weights using sample

===================================*/

function sample() {
  return Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());
}
