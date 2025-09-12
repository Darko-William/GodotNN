extends RichTextLabel

func get_X() -> Array:
	var csv = [] 
	var file = FileAccess.open("res://iris.csv", FileAccess.READ)
	while !file.eof_reached():
		var csv_rows = file.get_csv_line(",")
		var csv_num_rows = [csv_rows[0].to_float(), csv_rows[1].to_float(), csv_rows[2].to_float(), csv_rows[3].to_float()]
		csv.append(csv_num_rows)
	file.close()
	csv.pop_front()
	return csv
	
func get_y() -> Array:
	var csv = [] 
	var file = FileAccess.open("res://iris.csv", FileAccess.READ)
	while !file.eof_reached():
		var csv_rows = file.get_csv_line(",")
		var csv_num_rows = int(csv_rows[4])
		csv.append(csv_num_rows)
	file.close()
	csv.pop_front()
	return csv
	
func ReLU(ndarr):
	return nd.maximum(0, ndarr)

func ReLU_grad(ndarr):
	var grad = nd.zeros(ndarr.shape())
	for r in range(ndarr.shape()[0]):
		for c in range(ndarr.shape()[1]):
			if ndarr.get(r, c).to_int() > 0:
				grad.set(1, r, c)
	return grad

func softmax(ndarr):
	var smax := nd.zeros(ndarr.shape())
	for c in range(ndarr.shape()[1]):
		var sum = 0
		for r in range(ndarr.shape()[0]):
			sum += exp(ndarr.get(r, c).to_float())
		for r in range(ndarr.shape()[0]):
			var val = exp(ndarr.get(r, c).to_float()) / sum
			smax.set(val, r, c)
	return smax

func one_hot(Y: NDArray) -> NDArray:
	var one_hot_Y = []
	for i in Y:
		var enc = [0, 0, 0]
		enc[i.to_int()] = 1
		one_hot_Y.append(enc)
	return nd.array(one_hot_Y).transpose()

func init_params():
	# Init params
	var rand := NDRandomGenerator.new()
	var W1 := nd.add(rand.random([5, 4]), -0.5) # 5 neurons in hidden layer
	var b1 := nd.add(rand.random([5, 1]), -0.5)
	var W2 := nd.add(rand.random([3, 5]), -0.5)
	var b2 := nd.add(rand.random([3, 1]), -0.5)
	return [W1, b1, W2, b2]

func forward_prop(W1, b1, W2, b2, X):
	var Z1 := nd.add(nd.dot(W1, X), b1)
	var A1 = ReLU(Z1)
	var Z2 := nd.add(nd.dot(W2, A1), b2)
	var y_hat = softmax(Z2)
	return [Z1, A1, Z2, y_hat]

func back_prop(W1, b1, W2, b2, X, y, Z1, A1, Z2, y_hat):
	var one_hot_y := one_hot(y)
	var L := nd.divide(nd.sum(nd.pow(nd.subtract(y_hat, one_hot_y), 2), 0), 3) # Loss function: MSE
	# L = MSE(y_hat) = MSE(softmax(Z2)) = MSE(softmax(W2*A1 + b2))
	# dLdW2 = dLd(y_hat) * d(y_hat)dZ2 * dZ2dW2
	var dLdZ2:= nd.divide(nd.multiply(nd.subtract(y_hat, one_hot_y), 2), 2)
	# var dy_hatdZ2 
	var dZ2dW2 = A1.transpose()
	var dLdW2 = nd.dot(dLdZ2, dZ2dW2)
	var dZ2db2 = nd.ones([dLdZ2.shape()[1], 1])
	var dLdb2 = nd.dot(dLdZ2, dZ2db2)
	
	# dLdW1 = dLdZ2 * dZ2dA1 * dA1dZ1 * dZ1dW1
	var dZ2dA1 = W2.transpose()
	#var dA1dZ1 = ReLU_grad(Z1)
	var dZ1dW1 = X.transpose()
	var dLdW1 := nd.divide(nd.dot(nd.dot(dZ2dA1, dLdZ2), dZ1dW1), 3)
	
	# dLdb1 = dLdZ2 * dZ2dA1 * dA1dZ1 * dZ1db1
	var dZ1db1 = nd.ones([dLdZ2.shape()[1], 1])
	var dLdb1 = nd.divide(nd.dot(nd.dot(dZ2dA1, dLdZ2), dZ1db1), 3)
	return [dLdW1, dLdb1, dLdW2, dLdb2]

func update_params(W1, b1, W2, b2, dLdW1, dLdb1, dLdW2, dLdb2, lr) -> Array[NDArray]:
	var _W1 := nd.subtract(W1, nd.multiply(dLdW1, lr))
	var _b1 := nd.subtract(b1, nd.multiply(dLdb1, lr))
	var _W2 := nd.subtract(W2, nd.multiply(dLdW2, lr))
	var _b2 := nd.subtract(b2, nd.multiply(dLdb2, lr))
	return [_W1, _b1, _W2, _b2]

func _on_button_pressed() -> void:
	self.append_text("Trained Parameters: ")
	# Get Data
	var X := nd.array(get_X()).transpose()
	var y := nd.array(get_y())
	
	# Init Params
	var params = init_params()
	
	self.append_text("Layer 1: " + params[0].to_string())
	self.append_text("Layer 2: " + params[1].to_string())
	
	# Forward Propagation
	var forward = forward_prop(params[0], params[1], params[2], params[3], X)
	self.append_text(forward[0].to_string())
	
	# Back Propagation
	var grads = back_prop(params[0], params[1], params[2], params[3], X, y, forward[0], forward[1], forward[2], forward[3])
	self.append_text(grads[0].to_string())
	# Update Params
	var new_params := update_params(params[0], params[1], params[2], params[3], grads[0], grads[1], grads[2], grads[3], 0.1)
	var weights := new_params[0]
	
	self.append_text(weights.to_string())
