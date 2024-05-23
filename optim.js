"use strict"
var nj = require("numjs")


class Optim {

	constructor(params){
		this.params = params;
	}

	zeroGrad() {
		for (var param of this.params) {
			param.grad = null
		}
	}
}

class SGD extends Optim {

	constructor(params, lr=0.01, bs=1){
		super(params)
		this.lr = lr;
		this.bs = bs;
	}

	step() {
		for (var param of this.params){
			var batchGrad = nj.divide(param.grad, this.bs)
			var update = nj.multiply(batchGrad, this.lr)
			param.selection = nj.subtract(param.selection, update.T)
		}
	}
}

class Adam extends Optim {

	constructor(params, lr=0.001, bs=1, b1=0.9, b2=0.999, eps=1e-8){
		super(params)
		this.lr = lr
		this.bs = bs
		this.b1 = b1
		this.b2 = b2
		this.eps = eps
		this.t = 0
		this.m = []
		this.v = []

		for (var param of params) {
			this.m.push(nj.zeros(param.shape))
			this.v.push(nj.zeros(param.shape))
		}
	}

	step(){
		this.t += 1
		var a = this.lr * ((1.0 - this.b2**this.t)**0.5) / (1.0 - this.b1**this.t)
		
		for (let i=0; i<this.params.length; i++){
			var paramGrad = this.params[i].grad
			paramGrad = nj.divide(paramGrad, this.bs)

			var mComp = paramGrad.multiply((1 - this.b1))
			var vComp = paramGrad.multiply(paramGrad).multiply((1 - this.b2))

			this.m[i] = this.m[i].multiply(this.b1).add( mComp.T )
			this.v[i] = this.v[i].multiply(this.b2).add( vComp.T )
			
			var update = this.m[i].divide(this.v[i].sqrt().add(this.eps)).multiply(a)

			this.params[i].selection = nj.subtract(this.params[i].selection, update)
		}
	}
}

class StepLR {
	constructor(optimiser, stepSize=30, gamma=0.1, lastEpoch=-1){
		this.opt = optimiser;
		this.stepSize = stepSize;
		this.gamma = gamma;
		this.lastEpoch = lastEpoch;
	}

	step() {
		this.lastEpoch += 1
		if (this.lastEpoch % this.stepSize == 0 && this.lastEpoch != 0) {
			this.opt.lr *= this.gamma
		}
	}
}

module.exports = {
	SGD,
	Adam,
	StepLR
}