"use strict"
var nj = require("numjs");
var ops = require("./ops.js");
var tensorID = 0


var Tensor = function Tensor(value, params={op:null, parents:[], requiresGrad:false, label:null}) {
	this.selection = nj.array(value);
	// when the tensor is printed, the selection (array) is returned
	this[Symbol.for('nodejs.util.inspect.custom')] = () => this.selection;
	this.grad = null;
	this.shape = this.selection.shape
	
	this.tensorID = tensorID
	tensorID += 1

	this.op = (params.op) ? params.op : null
	this.parents = (params.parents) ? [...params.parents] : []
	this.requiresGrad = (params.requiresGrad) ? params.requiresGrad : false
	this.label = (params.label) ? params.label : null
}

Tensor.prototype.grad = this.grad;

Tensor.prototype.shape = this.shape;

Tensor.prototype.constructDAG = function(graph={nodes:[], edges:[]}) {
	var nodeID = this.tensorID
	var opName = (this.op) ? (this.op.constructor.name) : 'None'

	// if node not already visited
	if (!( graph.nodes.some(el => el.id == nodeID) ))
		graph.nodes.push({
			id:nodeID, 
			op:opName, 
			label:this.label, 
			shape:this.shape, 
			requiresGrad:this.requiresGrad
		})

	// iterate over parent nodes
	if (this.parents.length != 0) {
		for (var parent of this.parents) {
			graph.edges.push({from:parent.tensorID, to:nodeID})
			graph = parent.constructDAG(graph)
		}
	}
	return graph
}

Tensor.prototype.findAllNodes = function(nodes) {
	if (!nodes.includes(this)) {
		nodes.push(this)

		if (this.parents.length != 0) {
			for (var parent of this.parents) {
				nodes = parent.findAllNodes(nodes)
			}
		}
	}
	return nodes
}


Tensor.prototype.backward = function() {
	var nodes = this.findAllNodes([])

	for (var node of nodes) {
		if (node.grad == null) {
			node.grad = nj.ones(node.selection.shape)
		}

		if (node.parents.length != 0){
			var parent_grads = node.op.backward(...node.parents, node.grad)
			console.assert(Array.isArray(parent_grads), `Error: an op.backward() is not returning an Array`)

			for (let i=0; i < parent_grads.length; i++){
				if(node.parents[i].grad){
					node.parents[i].grad = nj.add(node.parents[i].grad, parent_grads[i])
				} else{
					node.parents[i].grad = parent_grads[i]
				}
			}
		}
	}

	for (var node of nodes) {
		if (node.requiresGrad == false) {node.grad = null}
	}
}

Tensor.prototype.tolist = function() {
	return this.selection.tolist();
}

Tensor.prototype.dot = function(a) {
	var op = new ops.MatMul()
	return new Tensor(op.forward(this.selection, a.selection), {op:op, parents:[this, a]});
}

Tensor.prototype.mul = function(a) {
	var op = new ops.Multiply()
	return new Tensor(op.forward(this.selection, a.selection), {op:op, parents:[this, a]});
}

Tensor.prototype.add = function(a) {
	var op = new ops.Add()
	return new Tensor(op.forward(this.selection, a.selection), {op:op, parents:[this, a]});
}

Tensor.prototype.sub = function(a) {
	var op = new ops.Sub()
	return new Tensor(op.forward(this.selection, a.selection), {op:op, parents:[this, a]});
}

Tensor.prototype.pow = function(a) {
	var op = new ops.Pow()
	return new Tensor(op.forward(this.selection, a.selection), {op:op, parents:[this, a]});
}

Tensor.prototype.max = function() {
	var op = new ops.Max()
	return new Tensor(op.forward(this.selection), {op:op, parents:[this]});
}

Tensor.prototype.min = function() {
	var op = new ops.Min()
	return new Tensor(op.forward(this.selection), {op:op, parents:[this]});
}

Tensor.prototype.sum = function() {
	var op = new ops.Sum()
	return new Tensor(op.forward(this.selection), {op:op, parents:[this]});
}

Tensor.prototype.exp = function() {
	var op = new ops.Exp()
	return new Tensor(op.forward(this.selection), {op:op, parents:[this]});
}

Tensor.prototype.negative = function() {
	var op = new ops.Negative()
	return new Tensor(op.forward(this.selection), {op:op, parents:[this]});
}

Tensor.prototype.log = function() {
	var op = new ops.Log()
	return new Tensor(op.forward(this.selection), {op:op, parents:[this]});
}

Tensor.prototype.transpose = function() {
	var op = new ops.Transpose()
	return new Tensor(op.forward(this.selection), {op:op, parents:[this]});
}

Tensor.prototype.identity = function() {
	var op = new ops.Identity()
	return new Tensor(op.forward(this.selection), {op:op, parents:[this]});
}

Tensor.prototype.reshape = function(...shape) {
	this.selection = this.selection.reshape(...shape)
	this.shape = this.selection.shape
}

Tensor.prototype.retainGrad = function() {
	this.requiresGrad = true
}

Tensor.prototype.detach = function() {
	this.parents = []
}


function tensor(data, params={}) {
	Object.assign(params, {requiresGrad: true});
	return new Tensor(data, params);
}

function ones(shape, params={}) {
	Object.assign(params, {requiresGrad: true});
	return new Tensor(nj.ones(shape), params);
}

function zeros(shape, params={}) {
	Object.assign(params, {requiresGrad: true});
	return new Tensor(nj.zeros(shape), params);
}

function randn(shape, params={}) {
	Object.assign(params, {requiresGrad: true});
	return new Tensor(nj.random(shape), params);
}

function arange(args, params={}) {
	Object.assign(params, {requiresGrad: true});
	return new Tensor(nj.arange(...args), params);
}

function eye(shape, params={}) {
	Object.assign(params, {requiresGrad: true});
	return new Tensor(nj.identity(shape), params);
}

function relu(a) {
	var op = new ops.ReLU()
	return new Tensor(op.forward(a.selection), {op:op, parents:[a]})
}

function relu6(a) {
	var op = new ops.ReLU6()
	return new Tensor(op.forward(a.selection), {op:op, parents:[a]})
}

function leakyRelu(a, negativeSlope) {
	var op = new ops.LeakyReLU(negativeSlope)
	return new Tensor(op.forward(a.selection), {op:op, parents:[a]})
}

function sigmoid(a) {
	var op = new ops.Sigmoid()
	return new Tensor(op.forward(a.selection), {op:op, parents:[a]})
}

function tanh(a) {
	var op = new ops.Tanh()
	return new Tensor(op.forward(a.selection), {op:op, parents:[a]})
}

function selu(a, alpha, lambda) {
	var op = new ops.SELU(alpha, lambda)
	return new Tensor(op.forward(a.selection), {op:op, parents:[a]})
}

function softmax(a) {
	var op = new ops.Softmax()
	return new Tensor(op.forward(a.selection), {op:op, parents:[a]})
}

module.exports = {
	tensor, 
	ones, 
	zeros,
	randn,
	arange,
	eye,
	relu,
	relu6,
	leakyRelu,
	sigmoid,
	tanh,
	selu,
	softmax
};
