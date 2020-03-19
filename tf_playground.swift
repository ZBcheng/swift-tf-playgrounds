import TensorFlow

struct MLP: Layer {
    var layer1 = Dense<Float>(inputSize: 2, outputSize: 10, activation: relu)
    var layer2 = Dense<Float>(inputSize: 10, outputSize: 1, activation: relu)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let h0 = layer1(input).withDerivative { print("∂L/∂layer1 =", $0) }
        return layer2(h0)
    }
}

var classifier = MLP()
let optimizer = SGD(for: classifier, learningRate: 0.02)

let x: Tensor<Float> = [[0, 0], [0, 1], [1, 0], [1, 1]]
let y: Tensor<Float> = [0, 1, 1, 0]

for _ in 0..<10 {
    let 𝛁model = gradient(at: classifier) { classifier -> Tensor<Float> in
        let ŷ = classifier(x).withDerivative { print("∂L/∂ŷ =", $0) }
        let loss = (ŷ - y).squared().mean()
        print("Loss: \(loss)")
        return loss
    }
    optimizer.update(&classifier, along: 𝛁model)
}